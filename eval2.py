#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用法示例：
python evaluate_protein_llm.py \
  --decoder_model_path ./re_llama \
  --projector_ckpt ./model/trained_projectors_stage1_test4k/pytorch_model.bin \
  --data_save_path ./data \
  --csv_path ./data/test.csv \
  --split test \
  --batch_size 2 \
  --use_name False \
  --save_csv ./results_eval.csv

python evaluate_protein_llm.py \
  --decoder_model_path ./re_llama \
  --projector_ckpt ./model/trained_projectors_stage1_test24_name_new_prompt/projectors_and_config_stage1.pt \
  --data_save_path ./data --csv_path ./data/test.csv --split test \
  --batch_size 2 --use_name --save_csv ./results_eval_test24_name_new_prompt.csv
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import contextlib
import torch
import pandas as pd
from functools import partial
from tqdm import tqdm
import evaluate

# 你项目里的模块
from dataset import ProteinDataset, custom_collate_fn
from model import ProteinFunctionLLM
from transformers import LlamaTokenizer

# 可选：在线 full name 编码器（评测时放 CPU，避免 DataLoader 多进程 + CUDA）
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class NameEncoder(nn.Module):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device="cpu"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.cache = {}  # 简单 CPU 缓存

    @torch.no_grad()
    def encode(self, texts):
        outs, miss_idx, miss_txt = [None]*len(texts), [], []
        for i, t in enumerate(texts):
            t = t or ""
            if t in self.cache:
                outs[i] = self.cache[t]
            else:
                miss_idx.append(i); miss_txt.append(t)
        if miss_txt:
            batch = self.tok(miss_txt, padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.enc(**batch).last_hidden_state               # [B,L,D]
            mask = batch["attention_mask"].unsqueeze(-1)          # [B,L,1]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # [B,D]
            pooled = pooled.unsqueeze(1)                          # [B,1,D]
            for j, i in enumerate(miss_idx):
                self.cache[miss_txt[j]] = pooled[j].detach().to("cpu")
                outs[i] = self.cache[miss_txt[j]]
        return torch.stack(outs, dim=0).to(self.device)           # [B,1,D]


# ---------- 构造 inputs_embeds 并生成 ----------
@torch.no_grad()
def generate_batch(model, batch, tokenizer, gen_kwargs):
    device = model.device
    dtype = next(model.decoder.parameters()).dtype

    # 1) 取 batch 张量
    struct = batch["struct_embeds"].to(device)
    seq    = batch["seq_embeds"].to(device)
    inst   = {k: v.to(device) for k, v in batch["instruction_tokens"].items()}

    # 2) 两路 projector + 融合 + mean-pool
    proj_s = model.structure_projector(struct)        # [B,Ls,D]
    proj_q = model.sequence_projector(seq)            # [B,Lq,D]
    fused  = proj_s + proj_q                          # [B,L,D]
    Hp     = fused.mean(dim=1, keepdim=True).to(dtype=dtype)  # [B,1,D]

    # 3) 可选：名字门控（仅当模型开关 is_add_name=True 且 batch 带有 name_embeds）
    if getattr(model, "is_add_name", False) and ("name_embeds" in batch):
        dec_dtype  = dtype
        gate_dtype = model.gate_fc.weight.dtype
        hname = batch["name_embeds"].to(device)
        hname = model.name_projector(hname).to(dtype=gate_dtype)   # [B,1,D] for gate
        gate  = torch.sigmoid(model.gate_fc(hname))                # [B,1,D]
        hname = hname.to(dtype=dec_dtype)
        gate  = gate.to(dtype=dec_dtype)
        alpha = float(getattr(model, "name_alpha", 1.0))
        Hp = Hp + alpha * gate * hname

    # 4) Instruction token → embedding
    input_ids = inst["input_ids"]                      # [B, L]
    inputs_embeds = model.decoder.get_input_embeddings()(input_ids).to(dtype=dtype)

    # 5) 向量化替换占位符 <protein> 为 Hp（多处占位符统一替换）
    ph_id = model.placeholder_token_id
    mask = (input_ids == ph_id).unsqueeze(-1)          # [B, L, 1]
    inputs_embeds = torch.where(mask, Hp.expand_as(inputs_embeds), inputs_embeds)

    # 6) 生成
    attn_mask = inst["attention_mask"]
    gen_ids = model.decoder.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        **gen_kwargs
    )
    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_model_path", type=str, required=True, help="重建后的 LLaMA 解码器路径")
    parser.add_argument("--projector_ckpt", type=str, default="", help="(可选) 训练好的 projector/门控权重路径")
    parser.add_argument("--data_save_path", type=str, required=True, help="./data")
    parser.add_argument("--csv_path", type=str, required=True, help="./data/test.csv")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_name", action="store_true", help="评测时是否启用 full name 门控分支")
    parser.add_argument("--name_model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--save_csv", type=str, default="./results_eval.csv")
    parser.add_argument("--seed", type=int, default=1234)
    # 生成参数（保守，先走确定性）
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = parser.parse_args()

    # 固定随机性（便于复现实验）
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Tokenizer =====
    tokenizer = LlamaTokenizer.from_pretrained(
        args.decoder_model_path, bos_token='<s>', eos_token='</s>', unk_token='<unk>'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    # 确保 <protein> 存在
    placeholder_token = "<protein>"
    tokenizer.add_special_tokens({'additional_special_tokens': [placeholder_token]})
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # ===== DataLoader =====
    # 注意：为避免 DataLoader worker 中使用 CUDA 导致的问题，这里评测时将 NameEncoder 放 CPU，
    # 且 use_name=True 时将 num_workers 设为 0（或改为预编码再查表的模式）。
    name_encoder = NameEncoder(args.name_model_name, device="cpu") if args.use_name else None
    collate = partial(
        custom_collate_fn,
        tokenizer=tokenizer,
        name_encoder=name_encoder,
        use_name=args.use_name,
        name_dropout_p=0.0
    )

    dataset = ProteinDataset(csv_path=args.csv_path, processed_dir=args.data_save_path, split=args.split)
    if args.use_name:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,              # 关键：避免在 worker 里跑编码器
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=max(1, (os.cpu_count() or 1) // 2),
            pin_memory=(device == "cuda"),
            persistent_workers=True if (os.cpu_count() and (os.cpu_count() // 2) > 0) else False,
        )

    # ===== Model =====
    config = {
        "decoder_model_path": args.decoder_model_path,
        "struct_embed_dim": 128,   # ← 按你的预处理维度改
        "seq_embed_dim": 2048,     # ← 按你的预处理维度改
        "device": device,
        "is_add_name": bool(args.use_name),
        "name_embed_dim": 768,     # 与 NameEncoder 的隐藏维度一致
    }
    model = ProteinFunctionLLM(config).to(device)
    # 词表扩展（因新增了 <protein>）
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.placeholder_token_id = placeholder_token_id

    # (可选) 加载 projector/门控权重
    if args.projector_ckpt and os.path.isfile(args.projector_ckpt):
        sd = torch.load(args.projector_ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    # 推理加速设置
    model.eval()
    try:
        model.decoder.eval()
        if hasattr(model.decoder, "gradient_checkpointing_disable"):
            model.decoder.gradient_checkpointing_disable()
        if hasattr(model.decoder, "config"):
            model.decoder.config.use_cache = True
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    # ===== 预测与保存 =====
    preds, refs = [], []
    gen_kwargs = dict(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if device == "cuda" else contextlib.nullcontext())
        with amp_ctx:
            for batch in tqdm(loader, desc="Generating", total=len(loader)):
                if batch is None:
                    continue
                batch_preds = generate_batch(model, batch, tokenizer, gen_kwargs)
                preds.extend(batch_preds)
                ans_ids = batch["answer_tokens"]["input_ids"]
                refs.extend(tokenizer.batch_decode(ans_ids, skip_special_tokens=True))

    # 保存结果
    df = pd.DataFrame({"generated": preds, "reference": refs})
    df.to_csv(args.save_csv, index=False, encoding="utf-8")
    print(f"[Saved] {args.save_csv}")

    # ===== 计算指标 =====
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    # BLEU 期望 references 为 List[List[str]]
    bleurefs = [[r] for r in refs]
    res_bleu = bleu.compute(predictions=preds, references=bleurefs)
    res_rouge = rouge.compute(predictions=preds, references=refs)  # 返回 rouge1/rouge2/rougeL/rougeLsum
    res_bert = bert.compute(
        predictions=preds,
        references=refs,
        model_type="dmis-lab/biobert-large-cased-v1.1",
        num_layers=24,
        device=("cuda:0" if device == "cuda" else "cpu"),
        batch_size=64,
        lang="en",
    )

    def avg(x): return sum(x)/len(x) if isinstance(x, (list, tuple)) and len(x) > 0 else float('nan')

    print("==== Metrics ====")
    print(f"BLEU: {res_bleu.get('bleu', res_bleu)}")
    print(f"ROUGE-1: {res_rouge.get('rouge1'):.4f}  ROUGE-2: {res_rouge.get('rouge2'):.4f}  ROUGE-L: {res_rouge.get('rougeL'):.4f}")
    print(f"BERTScore F1 (BioBERT): {avg(res_bert['f1']):.4f}")


if __name__ == "__main__":
    main()
