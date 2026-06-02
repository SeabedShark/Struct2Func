#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_and_eval_modal_experiments.py

"""

import os, math, time, random, json, contextlib
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup, AutoTokenizer, AutoModel
import evaluate

# 你的项目模块
from new_prompt.dataset_inst import ProteinDataset, custom_collate_fn
from model import ProteinFunctionLLM


CONFIG = {
    # --- 路径 ---
    "decoder_model_path": "./re_llama",
    "projector_ckpt": "",  # 可为空字符串表示不用初值
    "resume_from": "./modal_train_eval_test/w_o_name/epoch_004.pt",     # 续训检查点路径，为空表示从头开始
    # "resume_from": "",     # 续训检查点路径，为空表示从头开始
    "csv_train": "./data/train.csv",
    "csv_val":   "./data/eval.csv", 
    "csv_test":  "./data/test.csv",
    "processed_dir": "./data",
    "train_split": "train",
    "val_split": "eval",
    "test_split": "test",
    "output_dir": "./modal_train_eval_test",

    # --- 维度 ---
    "struct_embed_dim": 128,
    "seq_embed_dim": 2048,
    "name_embed_dim": 768,

    # --- 训练超参 ---
    "epochs": 7,
    "batch_size": 2,
    "num_workers": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "grad_accum_steps": 4,
    "max_grad_norm": 1.0,
    "fp16": True,                 # 若decoder是FP16则用AMP+GradScaler；若是BF16则走BF16 autocast且不用GradScaler
    "freeze_decoder": True,       # 冻结大部分 decoder，只微调顶层 + projector
    "tune_decoder_top_layers": 0, # 解冻顶层层数
    "name_dropout_p": 0.1,        # 训练时对 name 的dropout

    # --- 提前停止与日志 ---
    "early_stop_patience": 3,
    "early_stop_delta": 1e-5,
    "log_every": 50,
    "seed": 42,
    "ckpt_every_epoch": True,   # 每个 epoch 结束都存一份
    "ckpt_keep_last": 3,        # 只保留最近 N 个 epoch ckpt

    # --- 生成与评测 ---
    "gen_kwargs": dict(max_new_tokens=150, do_sample=False, temperature=1.0, top_p=1.0,
                       num_beams=5, early_stopping=True),
    "bertscore_model": "dmis-lab/biobert-large-cased-v1.1",
    "name_encoder_model": "allenai/scibert_scivocab_uncased",  # 评测阶段用
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


# -------- NameEncoder（评测阶段用） --------
class NameEncoder(nn.Module):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device="cpu"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def encode(self, texts):
        import math as _math
        clean = []
        for t in texts:
            if isinstance(t, str): s = t.strip()
            else:
                if t is None: s = ""
                elif isinstance(t, float) and _math.isnan(t): s = ""
                else: s = str(t).strip()
            clean.append(s)
        outs, miss_idx, miss_txt = [None]*len(clean), [], []
        for i, t in enumerate(clean):
            if t in self.cache: outs[i] = self.cache[t]
            else: miss_idx.append(i); miss_txt.append(t)
        if miss_txt:
            batch = self.tok(miss_txt, padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.enc(**batch).last_hidden_state            # [B,L,D]
            mask = batch["attention_mask"].unsqueeze(-1)       # [B,L,1]
            pooled = (h*mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # [B,D]
            pooled = pooled.unsqueeze(1)                       # [B,1,D]
            for j, i in enumerate(miss_idx):
                key = miss_txt[j]
                self.cache[key] = pooled[j].detach().to("cpu")
                outs[i] = self.cache[key]
        return torch.stack(outs, dim=0).to(self.device)        # [B,1,D]


# -------- 构造 decoder 输入（训练） --------
def build_decoder_inputs(model, tokenizer, batch, dec_dtype):
    """
    拼接 [instruction | answer],labels 仅监督 answer 段。
    并将 instruction 段中 <protein> 的 token embedding 替换为软提示 Hp。
    """
    device = next(model.parameters()).device
    inst = {k: v.to(device) for k, v in batch["instruction_tokens"].items()}
    ans  = {k: v.to(device) for k, v in batch["answer_tokens"].items()}

    # 拼接
    input_ids = torch.cat([inst["input_ids"], ans["input_ids"]], dim=1)
    attn_mask = torch.cat([inst["attention_mask"], ans["attention_mask"]], dim=1)

    # 基础嵌入
    inputs_embeds = model.decoder.get_input_embeddings()(input_ids).to(dtype=dec_dtype).clone()

    # 软提示 Hp（按开关融合）
    Hp = None
    parts = []
    if getattr(model, "enable_struct", True) and "struct_embeds" in batch:
        parts.append(model.structure_projector(batch["struct_embeds"].to(device).to(dtype=dec_dtype)))
    if getattr(model, "enable_seq", True) and "seq_embeds" in batch:
        parts.append(model.sequence_projector(batch["seq_embeds"].to(device).to(dtype=dec_dtype)))
    if len(parts) == 2:
        if parts[0].size(1) != parts[1].size(1):
            raise ValueError("Struct/Seq token length mismatch. Fix collate.")
        Hp = (parts[0] + parts[1]).mean(dim=1, keepdim=True)
    elif len(parts) == 1:
        Hp = parts[0].mean(dim=1, keepdim=True)

    if getattr(model, "enable_name", False) and ("name_embeds" in batch):
        name_w_dtype = next(model.name_projector.parameters()).dtype
        gate_w_dtype = next(model.gate_fc.parameters()).dtype
        hname = batch["name_embeds"].to(device).to(dtype=name_w_dtype)      # [B,1,Dn]
        hname = model.name_projector(hname)                                  # [B,1,D]
        gate  = torch.sigmoid(model.gate_fc(hname.to(dtype=gate_w_dtype)))   # [B,1,D]
        hname = hname.to(dtype=dec_dtype); gate = gate.to(dtype=dec_dtype)
        Hp = hname * gate if Hp is None else Hp + hname * gate

    if Hp is None:
        raise RuntimeError("All modalities disabled. Enable at least one.")
    Hp = Hp.to(dtype=dec_dtype)  # ★ dtype 对齐

    # 替换 instruction 段中的 <protein> 位置
    B, L_inst = inst["input_ids"].size()
    for b in range(B):
        pos = (inst["input_ids"][b] == model.placeholder_token_id).nonzero(as_tuple=False).view(-1)
        for p in pos:
            inputs_embeds[b, p, :] = Hp[b, 0, :]

    # labels：instruction 段-100；answer 段监督（PAD -> -100）
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, L_inst:] = ans["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs_embeds, attn_mask, labels


# -------- 验证（返回 token 平均 loss） --------
@torch.no_grad()
def evaluate_epoch(model, tok, loader, fp16_flag):
    model.eval()
    device = next(model.parameters()).device
    dec_dtype = next(model.decoder.parameters()).dtype
    total_loss, total_tokens = 0.0, 0

    IS_BF16 = (dec_dtype == torch.bfloat16)
    use_amp_ctx = (fp16_flag and torch.cuda.is_available()) or IS_BF16
    if use_amp_ctx:
        try:
            amp_ctx = torch.amp.autocast('cuda', dtype=(torch.float16 if (fp16_flag and not IS_BF16) else torch.bfloat16))
        except TypeError:
            amp_ctx = torch.cuda.amp.autocast(enabled=True)
    else:
        amp_ctx = contextlib.nullcontext()

    for batch in loader:
        if batch is None: continue
        batch = move_to_device(batch, device)
        inputs_embeds, attn_mask, labels = build_decoder_inputs(model, tok, batch, dec_dtype)
        with amp_ctx:
            out = model.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                labels=labels, use_cache=False)
            loss = out.loss
        valid = (labels != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid
    model.train()
    return float("inf") if total_tokens == 0 else total_loss / total_tokens


# -------- 训练单一配置 --------
def train_one_config(CFG, tag, flags, device, tok):
    save_dir = os.path.join(CFG["output_dir"], tag)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.csv")

    # DataLoaders（训练/验证：训练阶段不做 name 编码以提速）
    def make_loader(which, shuffle):
        csv_path = CFG["csv_train"] if which == "train" else CFG["csv_val"]
        split_str = CFG["train_split"] if which == "train" else CFG["val_split"] 
        coll = partial(
            custom_collate_fn,
            tokenizer=tok,
            name_encoder=None,                          # 训练阶段不做SciBERT编码（更快）
            use_name=bool(flags["enable_name"]),
            name_dropout_p=CFG["name_dropout_p"],
        )
        ds = ProteinDataset(csv_path=csv_path, processed_dir=CFG["processed_dir"], split=split_str) 
        return DataLoader(
            ds,
            batch_size=CFG["batch_size"],
            shuffle=shuffle,
            num_workers=CFG["num_workers"],
            collate_fn=coll,
            pin_memory=(device == "cuda"),
            persistent_workers=False
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)

    # Model
    mcfg = {
        "decoder_model_path": CFG["decoder_model_path"],
        "struct_embed_dim":   CFG["struct_embed_dim"],
        "seq_embed_dim":      CFG["seq_embed_dim"],
        "name_embed_dim":     CFG["name_embed_dim"],
        "device":             device,
        "is_add_name":  flags["enable_name"],  # 兼容老字段
        "enable_seq":   flags["enable_seq"],
        "enable_struct":flags["enable_struct"],
        "enable_name":  flags["enable_name"],
    }
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = ProteinFunctionLLM(mcfg).to(device)
    model.decoder.resize_token_embeddings(len(tok))
    model.placeholder_token_id = tok.convert_tokens_to_ids("<protein>")

    # 可选：载入 projector 初值
    if CFG["projector_ckpt"] and os.path.isfile(CFG["projector_ckpt"]):
        ckpt = torch.load(CFG["projector_ckpt"], map_location=device)
        if "structure_projector" in ckpt: model.structure_projector.load_state_dict(ckpt["structure_projector"], strict=False)
        if "sequence_projector"  in ckpt: model.sequence_projector.load_state_dict(ckpt["sequence_projector"], strict=False)
        if flags["enable_name"] and "name_projector" in ckpt: model.name_projector.load_state_dict(ckpt["name_projector"], strict=False)
        if flags["enable_name"] and "gate_fc" in ckpt:        model.gate_fc.load_state_dict(ckpt["gate_fc"], strict=False)

    # 冻结策略
    for p in model.decoder.parameters():
        p.requires_grad = False
    # 显式冻结 lm_head（若存在）
    if hasattr(model.decoder, "lm_head"):
        for p in model.decoder.lm_head.parameters():
            p.requires_grad = False
    # 显式冻结输入词嵌入（不同实现下的路径稍有差异）
    emb_mod = getattr(getattr(model.decoder, "model", model.decoder), "embed_tokens", None)
    if emb_mod is not None:
        for p in emb_mod.parameters():
            p.requires_grad = False

    # 仅启用 projector 参数
    train_params = []
    for p in model.structure_projector.parameters():
        p.requires_grad = True
    for p in model.sequence_projector.parameters():
        p.requires_grad = True
    train_params += list(model.structure_projector.parameters()) + list(model.sequence_projector.parameters())

    # 若启用 name 分支，也只训练 name_projector 与 gate_fc
    if flags.get("enable_name", False):
        if hasattr(model, "name_projector"):
            for p in model.name_projector.parameters():
                p.requires_grad = True
            train_params += list(model.name_projector.parameters())
        if hasattr(model, "gate_fc"):
            for p in model.gate_fc.parameters():
                p.requires_grad = True
            train_params += list(model.gate_fc.parameters())

    # 调试打印：确认只有 projector 在训
    n_trainable_total = count_parameters(model)
    n_trainable_dec = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"[{tag}] Trainable params (TOTAL) = {n_trainable_total:,}")
    print(f"[{tag}] Trainable params (decoder) = {n_trainable_dec:,}  (should be 0)")


    # 优化器 / 调度器
    optimizer = torch.optim.AdamW(
        train_params, lr=CFG["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=CFG["weight_decay"]
    )
    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / CFG["grad_accum_steps"]))
    t_total      = CFG["epochs"] * num_update_steps_per_epoch
    warmup_steps = int(CFG["warmup_ratio"] * t_total)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # 续训逻辑
    start_epoch = 1
    step_global = 0
    patience = 0
    if CFG["resume_from"] and os.path.isfile(CFG["resume_from"]):
        print(f"[{tag}] 从检查点恢复训练: {CFG['resume_from']}")
        
        # 先清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 加载检查点到CPU，减少GPU内存压力
        ckpt = torch.load(CFG["resume_from"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        
        # 为了节省内存，续训时不恢复优化器和调度器状态
        # 只恢复模型权重和训练进度，优化器和调度器重新初始化
        print(f"[{tag}] 续训模式：重新初始化优化器和调度器（节省内存）")
            
        start_epoch = ckpt.get("epoch", 1) + 1  # 从下一个epoch开始
        step_global = ckpt.get("step_global", 0)
        patience = ckpt.get("patience", 0)
        print(f"[{tag}] 恢复状态: epoch={start_epoch}, step={step_global}, patience={patience}")
        
        # 清理检查点数据，释放内存
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # AMP / DTYPE 自适应
    dec_dtype = next(model.decoder.parameters()).dtype
    IS_BF16     = (dec_dtype == torch.bfloat16)
    AMP_ENABLED = (CONFIG["fp16"] and torch.cuda.is_available() and (not IS_BF16))
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_val   = float("inf")
    best_path  = os.path.join(save_dir, "best.pt")
    header_written = False
    model.train()

    # 初始化 epoch，避免跳过训练时 UnboundLocalError
    epoch = start_epoch - 1  # 如果循环被跳过，使用上一个 epoch
    
    # 训练循环
    for epoch in range(start_epoch, CFG["epochs"] + 1):
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}/{CFG['epochs']}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            if batch is None:
                print("one batch is none")
                continue
                
            batch = move_to_device(batch, device)
            dec_dtype = next(model.decoder.parameters()).dtype
            inputs_embeds, attn_mask, labels = build_decoder_inputs(model, tok, batch, dec_dtype)

            # 选择 autocast 上下文：FP16 用 float16，BF16 用 bfloat16；否则不开
            use_amp_ctx = AMP_ENABLED or IS_BF16
            if use_amp_ctx:
                try:
                    amp_ctx = torch.amp.autocast('cuda', dtype=(torch.float16 if AMP_ENABLED else torch.bfloat16))
                except TypeError:
                    amp_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                amp_ctx = contextlib.nullcontext()

            with amp_ctx:
                out  = model.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                     labels=labels, use_cache=False)
                loss = out.loss / CFG["grad_accum_steps"]

            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % CFG["grad_accum_steps"] == 0:
                if AMP_ENABLED:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                if CFG["max_grad_norm"] > 0:
                    clip_grad_norm_(train_params, CFG["max_grad_norm"])

                if AMP_ENABLED:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step_global += 1

            if step_global % CFG["log_every"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item()*CFG['grad_accum_steps']:.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # 验证 + 记录
        val_loss = evaluate_epoch(model, tok, val_loader, fp16_flag=CFG["fp16"])
        row = {"epoch": epoch, "val_loss/token": val_loss, "step": step_global, "lr": scheduler.get_last_lr()[0]}
        pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=not header_written)
        header_written = True
        print(f"[{tag}] Epoch {epoch} | val_loss/token={val_loss:.6f}")

        # 1) 保存 best.pt
        if val_loss + CFG["early_stop_delta"] < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": mcfg,
                "tokenizer_len": len(tok),
                "placeholder_token_id": model.placeholder_token_id,
                "flags": flags,
                "best_val": best_val,
                # 续训所需的状态
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "step_global": step_global,
                "patience": patience,
            }, best_path)
            print(f"[{tag}] Saved best to: {best_path}")
            patience = 0
        else:
            patience += 1

        # 2) 每 epoch 另存一份，保留最近 N 份
        if CFG.get("ckpt_every_epoch", False):
            epoch_ckpt = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "config": mcfg,
                "tokenizer_len": len(tok),
                "placeholder_token_id": model.placeholder_token_id,
                "flags": flags,
                "val_loss": val_loss,
                "epoch": epoch,
                # 续训所需的状态
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step_global": step_global,
                "patience": patience,
            }, epoch_ckpt)
            print(f"[{tag}]  Saved epoch ckpt: {epoch_ckpt}")

            k = int(CFG.get("ckpt_keep_last", 0))
            if k > 0:
                eps = sorted([f for f in os.listdir(save_dir) if f.startswith("epoch_") and f.endswith(".pt")])
                to_rm = eps[:-k]  # 保留最后 k 个
                for fn in to_rm:
                    try:
                        os.remove(os.path.join(save_dir, fn))
                    except Exception as e:
                        print(f"[{tag}] warn: remove {fn} failed: {e}")

        # 3) 早停
        if patience >= CFG["early_stop_patience"]:
            print(f"[{tag}]  Early stop (patience={CFG['early_stop_patience']})")
            break

    # 保存 last.pt
    last_path = os.path.join(save_dir, "last.pt")
    torch.save({
        "model": model.state_dict(), 
        "config": mcfg, 
        "flags": flags,
        # 续训所需的状态
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_global": step_global,
        "patience": patience,
    }, last_path)
    print(f"[{tag}] Finished training. Best={best_val:.6f}")

    try:
        del train_loader, val_loader
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_path


# -------- 评测：加载 best 生成 --------
@torch.no_grad()
def generate_with_modalities(model: ProteinFunctionLLM, batch, tokenizer, gen_kwargs):
    device = next(model.parameters()).device
    dec_dtype = next(model.decoder.parameters()).dtype

    inst = {k: v.to(device) for k, v in batch["instruction_tokens"].items()}
    input_ids = inst["input_ids"]
    inputs_embeds = model.decoder.get_input_embeddings()(input_ids).to(dtype=dec_dtype).clone()

    # 软提示
    Hp = None
    parts = []
    if getattr(model, "enable_struct", True) and "struct_embeds" in batch:
        parts.append(model.structure_projector(batch["struct_embeds"].to(device).to(dtype=dec_dtype)))
    if getattr(model, "enable_seq", True) and "seq_embeds" in batch:
        parts.append(model.sequence_projector(batch["seq_embeds"].to(device).to(dtype=dec_dtype)))
    if len(parts) == 2:
        if parts[0].size(1) != parts[1].size(1): return None
        Hp = (parts[0] + parts[1]).mean(dim=1, keepdim=True)
    elif len(parts) == 1:
        Hp = parts[0].mean(dim=1, keepdim=True)

    if getattr(model, "enable_name", False) and ("name_embeds" in batch):
        name_w_dtype = next(model.name_projector.parameters()).dtype
        gate_w_dtype = next(model.gate_fc.parameters()).dtype
        hname = batch["name_embeds"].to(device).to(dtype=name_w_dtype)
        hname = model.name_projector(hname)
        gate  = torch.sigmoid(model.gate_fc(hname.to(dtype=gate_w_dtype)))
        hname = hname.to(dtype=dec_dtype); gate = gate.to(dtype=dec_dtype)
        Hp = hname * gate if Hp is None else Hp + hname * gate
    if Hp is None: raise RuntimeError("No modality enabled.")

    Hp = Hp.to(dtype=dec_dtype)  # ★ dtype 对齐

    # 替换 <protein>
    for b in range(inputs_embeds.size(0)):
        pos = (input_ids[b] == model.placeholder_token_id).nonzero(as_tuple=False).view(-1)
        for p in pos:
            inputs_embeds[b, p, :] = Hp[b, 0, :]

    gen_ids = model.decoder.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inst["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs
    )
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def eval_one_config(CFG, tag, best_path, device, tok):
    save_dir = os.path.join(CFG["output_dir"], tag)
    out_csv = os.path.join(save_dir, f"{tag}.test.csv")

    # 载入 best
    ckpt = torch.load(best_path, map_location=device)
    flags = ckpt.get("flags", {"enable_seq": True, "enable_struct": True, "enable_name": False})
    mcfg  = ckpt["config"]

    model = ProteinFunctionLLM(mcfg).to(device)
    model.decoder.resize_token_embeddings(len(tok))
    model.load_state_dict(ckpt["model"], strict=False)
    model.placeholder_token_id = ckpt.get("placeholder_token_id", tok.convert_tokens_to_ids("<protein>"))
    model.eval()

    # 测试集 DataLoader（评测阶段需要 name 编码）
    name_enc = NameEncoder(CONFIG["name_encoder_model"], device="cpu") if flags["enable_name"] else None
    coll = partial(custom_collate_fn, tokenizer=tok, name_encoder=name_enc,
                   use_name=bool(flags["enable_name"]), name_dropout_p=0.0)

    ds_test = ProteinDataset(csv_path=CONFIG["csv_test"],
                             processed_dir=CONFIG["processed_dir"],
                             split=CONFIG["test_split"])
    loader = DataLoader(ds_test, batch_size=CONFIG["batch_size"], shuffle=False,
                        collate_fn=coll, num_workers=0, pin_memory=(device=="cuda"))

    preds, refs, accs = [], [], []
    for batch in tqdm(loader, desc=f"[{tag}] Generating[Test]"):
        if batch is None: continue
        out = generate_with_modalities(model, batch, tok, CFG["gen_kwargs"])
        if out is None: continue
        preds.extend(out)
        refs.extend(tok.batch_decode(batch["answer_tokens"]["input_ids"], skip_special_tokens=True))
        accs.extend(batch.get("accession_list", [""]*len(out)))

    pd.DataFrame({"accession": accs, "generated": preds, "reference": refs}).to_csv(out_csv, index=False, quoting=1)
    print(f"[{tag}]  Saved test outputs: {out_csv}")

    # 评测指标
    bleu  = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert  = evaluate.load("bertscore")
    res_bleu  = bleu.compute(predictions=preds, references=refs)
    res_rouge = rouge.compute(predictions=preds, references=refs)
    res_bert  = bert.compute(predictions=preds, references=refs,
                             model_type=CFG["bertscore_model"],
                             num_layers=24, 
                             )

    def _avg(x): return float(sum(x)/len(x)) if isinstance(x,(list,tuple)) and len(x)>0 else float("nan")
    row = {
        "tag": tag,
        "N": len(preds),
        "BLEU": float(res_bleu.get("bleu", res_bleu.get("score", float("nan")))),
        "ROUGE-1": float(res_rouge.get("rouge1", float("nan"))),
        "ROUGE-2": float(res_rouge.get("rouge2", float("nan"))),
        "ROUGE-L": float(res_rouge.get("rougeL", float("nan"))),
        "BERTScore_F1": _avg(res_bert.get("f1", [])),
    }
    pd.DataFrame([row]).to_csv(os.path.join(save_dir, "test.metrics.csv"), index=False)
    print(f"[{tag}]  Metrics: {row}")
    return row


def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer：左填充 + 注入 <protein>
    tok = LlamaTokenizer.from_pretrained(CONFIG["decoder_model_path"],
                                         bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
        tok.pad_token_id = tok.unk_token_id
    tok.padding_side = "left"
    tok.add_special_tokens({'additional_special_tokens': ["<protein>"]})

    # 六种配置（暂只跑 only_seq，按需放开）
    experiments = [
        # ("only_seq",    dict(enable_seq=True,  enable_struct=False, enable_name=False)),
        # ("only_struct", dict(enable_seq=False, enable_struct=True,  enable_name=False)),
        # ("only_name",   dict(enable_seq=False, enable_struct=False, enable_name=True)),
        # ("w_o_seq",     dict(enable_seq=False, enable_struct=True,  enable_name=True)),
        # ("w_o_struct",  dict(enable_seq=True,  enable_struct=False, enable_name=True)),
        ("w_o_name",    dict(enable_seq=True,  enable_struct=True,  enable_name=False)),
    ]

    all_rows = []
    for tag, flags in experiments:
        print("\n" + "="*86)
        print(f"==> Training: {tag} | flags={flags}")
        print("="*86)
        best_path = train_one_config(CONFIG, tag, flags, device, tok)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n==> Evaluating: {tag}")
        row = eval_one_config(CONFIG, tag, best_path, device, tok)
        all_rows.append(row)

    # 汇总表
    summary_path = os.path.join(CONFIG["output_dir"], "summary_metrics.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nAll done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
