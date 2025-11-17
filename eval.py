#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
python evaluate_protein_llm.py \
  --decoder_model_path ./re_llama \
  --projector_ckpt ./model/trained_projectors_stage1_test4k/pytorch_model.bin \
  --data_save_path ./data \
  --csv_path ./data/test.csv \
  --split test \
  --batch_size 2 \
  --use_name False \
  --save_csv ./results_eval.csv
  --score_only

python eval.py --decoder_model_path ./re_llama --projector_ckpt ./model/trained_projectors_stage1_test24_name_new_prompt/projectors_and_config_stage1.pt --data_save_path ./data --csv_path ./data/test.csv --split test --batch_size 2 --use_name --save_csv ./results_eval_test24_name_new_prompt.csv
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import torch
import pandas as pd
from functools import partial
from tqdm import tqdm
import evaluate


from new_prompt.dataset_inst import ProteinDataset, custom_collate_fn
import numpy as np
import random
from model import ProteinFunctionLLM
from transformers import LlamaTokenizer


from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class NameEncoder(nn.Module):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device="cuda"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.cache = {}  # 简单 CPU 缓存，跨 batch/epoch 复用

    @torch.no_grad()
    def encode(self, texts):
        # 预清洗：将 None/NaN/非字符串安全转换为短字符串
        clean_texts = []
        for t in texts:
            if isinstance(t, str):
                s = t.strip()
            else:
                if t is None:
                    s = ""
                elif isinstance(t, float) and math.isnan(t):
                    s = ""
                else:
                    s = str(t).strip()
            clean_texts.append(s)

        outs, miss_idx, miss_txt = [None]*len(clean_texts), [], []
        for i, t in enumerate(clean_texts):
            if t in self.cache:
                outs[i] = self.cache[t]
            else:
                miss_idx.append(i); miss_txt.append(t)
        if miss_txt:
            batch = self.tok(miss_txt, padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.enc(**batch).last_hidden_state               # [B,L,D]
            mask = batch["attention_mask"].unsqueeze(-1)          # [B,L,1]
            pooled = (h*mask).sum(1) / mask.sum(1).clamp_min(1e-6)# [B,D]
            pooled = pooled.unsqueeze(1)                          # [B,1,D]
            for j, i in enumerate(miss_idx):
                key = miss_txt[j]
                self.cache[key] = pooled[j].detach().to("cpu")
                outs[i] = self.cache[key]
        return torch.stack(outs, dim=0).to(self.device)           # [B,1,D]

# ---------- 构造 inputs_embeds 并生成 ----------
@torch.no_grad()
def generate_batch(model, batch, tokenizer, gen_kwargs):
   
    device = model.device
    dtype = next(model.decoder.parameters()).dtype

    # 1) 取 batch 张量
    struct = batch["struct_embeds"].to(device).to(dtype)
    seq   = batch["seq_embeds"].to(device).to(dtype)
    inst  = {k: v.to(device) for k, v in batch["instruction_tokens"].items()}

    # 2) 两路 projector + 检查长度一致性 + 融合 + mean-pool
    proj_s = model.structure_projector(struct)        # [B,Ls,D]
    proj_q = model.sequence_projector(seq)            # [B,Lq,D]
    if proj_s.size(1) != proj_q.size(1):
        # 在 eval 中对不一致样本直接跳过该 batch
        return None
    fused  = proj_s + proj_q                          # [B,L,D]
    Hp     = fused.mean(dim=1, keepdim=True)          # [B,1,D]
    Hp     = Hp.to(dtype=dtype)

    # 3) 可选：名字门控（仅当模型开关 is_add_name=True 且 batch 带有 name_embeds）
    if getattr(model, "is_add_name", False) and ("name_embeds" in batch):
        # 统一到解码器 dtype，避免 Float/BFloat16 混用
        dec_dtype = dtype
        hname = batch["name_embeds"].to(device).to(dec_dtype)
        hname = model.name_projector(hname)                         # [B,1,D] in dec_dtype
        gate  = torch.sigmoid(model.gate_fc(hname))                 # [B,1,D] in dec_dtype
        alpha = float(getattr(model, "name_alpha", 1.0))
        Hp = Hp + alpha * gate * hname

    # 4) Instruction token → embedding
    input_ids = inst["input_ids"]                                # [B,Lin]
    inputs_embeds = model.decoder.get_input_embeddings()(input_ids).to(dtype=dtype)

    # 5) 替换占位符 <protein> 的位置为 Hp
    ph_id = model.placeholder_token_id
    B, Lin, D = inputs_embeds.shape
    inputs_embeds = inputs_embeds.clone()
    for b in range(B):
        pos = (input_ids[b] == ph_id).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0:
            continue
        # 若出现多个 <protein>，都替换为同一个 pooled Hp[b,0]
        for p in pos:
            inputs_embeds[b, p, :] = Hp[b, 0, :]

    # 6) 生成
    attn_mask = inst["attention_mask"]
    gen_ids = model.decoder.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs
    )
    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return preds

def main():
    # 内置配置参数，去掉命令行依赖
    CONFIG = {
        "decoder_model_path": "./re_llama",
        "projector_ckpt": "./modal_train_eval_test/only_name/epoch_004.pt",
        "data_save_path": "./data",
        "csv_path": "./data/test.csv",
        "split": "test",
        "batch_size": 2,
        "use_name": True,
        "name_model_name": "allenai/scibert_scivocab_uncased",
        # "save_csv": "./train_11_epochs_text_results_eval.csv",
        "save_csv": "./onlyname_4epoch.csv",
        # 评测模式：若已存在 save_csv 则直接读并评分
        "score_only": False,
        # 生成策略：与 g_inst.py 对齐（确定性、高质量）
        "gen": {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "num_beams": 5,
            "early_stopping": True,
            # 常用防重复项（可选）
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        },
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 设定随机种子，保证与 g_inst 一致的可复现性
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
    except Exception:
        pass

    # ===== 仅评分分支：直接读取已有CSV并计算指标 =====
    if CONFIG.get("score_only", False):
        if not os.path.isfile(CONFIG["save_csv"]):
            raise FileNotFoundError(f"score_only 模式下未找到结果CSV: {CONFIG['save_csv']}")
        df_exist = pd.read_csv(CONFIG["save_csv"])
        # 期望包含 generated/reference 两列
        if not {"generated", "reference"}.issubset(set(df_exist.columns)):
            raise ValueError(f"CSV 缺少必要列: generated/reference; 实际列: {list(df_exist.columns)}")
        preds = df_exist["generated"].astype(str).tolist()
        refs = df_exist["reference"].astype(str).tolist()

        # ===== 计算指标 =====
        bleu = evaluate.load("bleu")        # 与参考脚本一致
        rouge = evaluate.load("rouge")
        bert = evaluate.load("bertscore")

        res_bleu = bleu.compute(predictions=preds, references=refs)
        res_rouge = rouge.compute(predictions=preds, references=refs)
        res_bert = bert.compute(predictions=preds, references=refs,
                                model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24)

        def avg(x): return sum(x)/len(x) if isinstance(x, (list, tuple)) and len(x)>0 else float('nan')

        print("==== Metrics (score_only) ====")
        print(f"BLEU: {res_bleu.get('bleu', res_bleu)}")
        print(f"ROUGE-1: {res_rouge.get('rouge1'):.4f}  ROUGE-2: {res_rouge.get('rouge2'):.4f}  ROUGE-L: {res_rouge.get('rougeL'):.4f}")
        print(f"BERTScore F1 (BioBERT): {avg(res_bert['f1']):.4f}")

        # 追加写入 metrics CSV（与正常流程保持一致字段）
        try:
            import datetime as _dt
            metrics_path = f"{CONFIG['save_csv']}.metrics.csv"
            bleu_value = res_bleu.get('bleu', res_bleu.get('score', float('nan'))) if isinstance(res_bleu, dict) else res_bleu
            row = {
                "projector_ckpt": CONFIG["projector_ckpt"],
                "use_name": bool(CONFIG["use_name"]),
                "num_samples": len(preds),
                "do_sample": bool(CONFIG["gen"].get("do_sample", False)),
                "num_beams": int(CONFIG["gen"].get("num_beams", 1)),
                "BLEU": bleu_value,
                "ROUGE-1": float(res_rouge.get('rouge1', float('nan'))),
                "ROUGE-2": float(res_rouge.get('rouge2', float('nan'))),
                "ROUGE-L": float(res_rouge.get('rougeL', float('nan'))),
                "BERTScore_F1": float(avg(res_bert.get('f1', [])))
            }
            mdf = pd.DataFrame([row])
            write_header = not os.path.exists(metrics_path)
            mdf.to_csv(metrics_path, mode='a', index=False, header=write_header)
            print(f"[Metrics appended] {metrics_path}")
        except Exception as e:
            print(f"[Metrics CSV write skipped] {e}")
        return

    # ===== Tokenizer =====
    tokenizer = LlamaTokenizer.from_pretrained(CONFIG["decoder_model_path"],
                                               bos_token='<s>', eos_token='</s>', unk_token='<unk>')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    # 确保 <protein> 存在
    placeholder_token = "<protein>"
    tokenizer.add_special_tokens({'additional_special_tokens': [placeholder_token]})
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # ===== DataLoader =====
    # 若你训练时就在 collate 里在线编码名字，这里同样传入 name_encoder；否则 use_name 设为 False 即可
    # 注意：为避免多进程 DataLoader + CUDA 在 worker 中初始化导致的错误，NameEncoder 放在 CPU 上执行
    name_encoder = NameEncoder(CONFIG["name_model_name"], device="cpu") if CONFIG["use_name"] else None
    collate = partial(custom_collate_fn, tokenizer=tokenizer,
                      name_encoder=name_encoder, use_name=CONFIG["use_name"], name_dropout_p=0.0)

    dataset = ProteinDataset(csv_path=CONFIG["csv_path"], processed_dir=CONFIG["data_save_path"], split=CONFIG["split"])
    if CONFIG["use_name"]:
        # 避免在 worker 进程里使用 CUDA：worker 设为 0
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=max(1, os.cpu_count() // 2),
            pin_memory=(device == "cuda"),
            persistent_workers=True if (os.cpu_count() and os.cpu_count() // 2 > 0) else False,
        )

    # ===== Model =====
    config = {
        "decoder_model_path": CONFIG["decoder_model_path"],
        "struct_embed_dim": 128,   # ← 按你的预处理维度改
        "seq_embed_dim": 2048,     # ← 按你的预处理维度改
        "device": device,
        "is_add_name": bool(CONFIG["use_name"]),
        "name_embed_dim": 768,     # 与 NameEncoder 的隐藏维度一致
    }
    model = ProteinFunctionLLM(config).to(device)
    # 词表扩展（因新增了 <protein>）
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.placeholder_token_id = placeholder_token_id
    # 与 g_inst 一致地加载 projector 权重并对齐 dtype
    if CONFIG["projector_ckpt"] and os.path.isfile(CONFIG["projector_ckpt"]):
        try:
            ckpt = torch.load(CONFIG["projector_ckpt"], map_location=device)
            # 仅提取需要的子模块权重
            if "structure_projector" in ckpt:
                model.structure_projector.load_state_dict(ckpt["structure_projector"])
            if "sequence_projector" in ckpt:
                model.sequence_projector.load_state_dict(ckpt["sequence_projector"])
            # 可选的名字分支
            if getattr(model, "is_add_name", False):
                if "name_projector" in ckpt:
                    model.name_projector.load_state_dict(ckpt["name_projector"])
                if "gate_fc" in ckpt:
                    model.gate_fc.load_state_dict(ckpt["gate_fc"])
            # 将 projector / gate 对齐到解码器 dtype
            dec_dtype = next(model.decoder.parameters()).dtype
            model.structure_projector = model.structure_projector.to(dtype=dec_dtype, device=device)
            model.sequence_projector = model.sequence_projector.to(dtype=dec_dtype, device=device)
            if getattr(model, "is_add_name", False):
                model.name_projector = model.name_projector.to(dtype=dec_dtype, device=device)
                model.gate_fc = model.gate_fc.to(dtype=dec_dtype, device=device)
        except Exception as e:
            print(f"[Warning] Failed to load projector ckpt in eval: {e}")

    # 调试信息：确认与 g_inst 对齐
    try:
        print("==== Eval Alignment Check ====")
        print(f"decoder_model_path: {CONFIG['decoder_model_path']}")
        print(f"projector_ckpt: {CONFIG['projector_ckpt']}")
        print(f"is_add_name: {bool(CONFIG.get('use_name', False))}")
        print(f"decoder dtype: {next(model.decoder.parameters()).dtype}")
        print(f"structure_projector dtype: {next(model.structure_projector.parameters()).dtype}")
        print(f"sequence_projector dtype: {next(model.sequence_projector.parameters()).dtype}")
        if getattr(model, 'is_add_name', False):
            print(f"name_projector dtype: {next(model.name_projector.parameters()).dtype}")
            print(f"gate_fc dtype: {model.gate_fc.weight.dtype}")
        print(f"placeholder_token_id: {placeholder_token_id}")
    except Exception:
        pass

    model.eval()
    #启用KV缓存，禁用梯度检查点，并允许TF32
    try:
        model.decoder.eval()
        if hasattr(model.decoder, "gradient_checkpointing_disable"):
            model.decoder.gradient_checkpointing_disable()
        if hasattr(model.decoder, "config"):
            model.decoder.config.use_cache = True
        torch.set_grad_enabled(False)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    # ===== 预测与保存 =====
    preds, refs = [], []
    accessions = []  # 保存 accession
    names = []  # 若 CSV 有 protein 名称列可以带上
    # 生成参数严格对齐 g_inst：确定性 beam search
    gen_kwargs = dict(
        max_new_tokens=150,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        num_beams=5,
        early_stopping=True,
    )
    print(f"gen_kwargs: {gen_kwargs}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            if batch is None:  # 被过滤的空 batch
                continue
            # 预测
            batch_preds = generate_batch(model, batch, tokenizer, gen_kwargs)
            # 打印当前 batch 的生成结果
            try:
                print("[batch_preds]", batch_preds)
            except Exception:
                pass
            preds.extend(batch_preds)
            # 参考（数据集里 answer_tokens 已经是你构造的功能描述）
            ans_ids = batch["answer_tokens"]["input_ids"]
            batch_refs = tokenizer.batch_decode(ans_ids, skip_special_tokens=True)
            refs.extend(batch_refs)
            # accession 列
            if "accession_list" in batch:
                accessions.extend(batch["accession_list"])
            # 可选：名字
            # names.extend(batch.get("name_list", [""]*len(batch_preds)))

    # 保存结果
    df = pd.DataFrame({"accession": accessions, "generated": preds, "reference": refs})
    df.to_csv(CONFIG["save_csv"], index=False,quoting=1)
    print(f"[Saved] {CONFIG['save_csv']}")

    # ===== 计算指标 =====
    bleu = evaluate.load("bleu")        # 与参考脚本一致
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    res_bleu = bleu.compute(predictions=preds, references=refs)
    res_rouge = rouge.compute(predictions=preds, references=refs)  # 返回 rouge1/rouge2/rougeL/rougeLsum
    res_bert = bert.compute(predictions=preds, references=refs,
                            model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24)

    def avg(x): return sum(x)/len(x) if isinstance(x, (list, tuple)) and len(x)>0 else float('nan')

    print("==== Metrics ====")
    print(f"BLEU: {res_bleu.get('bleu', res_bleu)}")
    print(f"ROUGE-1: {res_rouge.get('rouge1'):.4f}  ROUGE-2: {res_rouge.get('rouge2'):.4f}  ROUGE-L: {res_rouge.get('rougeL'):.4f}")
    print(f"BERTScore F1 (BioBERT): {avg(res_bert['f1']):.4f}")

    # ===== 追加写入汇总指标到 metrics CSV =====
    try:
        import datetime as _dt
        metrics_path = f"{CONFIG['save_csv']}.metrics.csv"
        bleu_value = res_bleu.get('bleu', res_bleu.get('score', float('nan'))) if isinstance(res_bleu, dict) else res_bleu
        row = {
            "projector_ckpt": CONFIG["projector_ckpt"],
            "use_name": bool(CONFIG["use_name"]),
            "num_samples": len(preds),
            "do_sample": bool(CONFIG["gen"].get("do_sample", False)),
            "num_beams": int(CONFIG["gen"].get("num_beams", 1)),
            "BLEU": bleu_value,
            "ROUGE-1": float(res_rouge.get('rouge1', float('nan'))),
            "ROUGE-2": float(res_rouge.get('rouge2', float('nan'))),
            "ROUGE-L": float(res_rouge.get('rougeL', float('nan'))),
            "BERTScore_F1": float(avg(res_bert.get('f1', [])))
        }
        mdf = pd.DataFrame([row])
        write_header = not os.path.exists(metrics_path)
        mdf.to_csv(metrics_path, mode='a', index=False, header=write_header)
        print(f"[Metrics appended] {metrics_path}")
    except Exception as e:
        print(f"[Metrics CSV write skipped] {e}")

if __name__ == "__main__":
    main()
