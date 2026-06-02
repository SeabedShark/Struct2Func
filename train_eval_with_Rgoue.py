import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler,LlamaTokenizer,AutoModel
from accelerate import Accelerator
from functools import partial
from tqdm import tqdm
import copy
import json
import gc
import random
import numpy as np
import time 
import evaluate

from new_prompt.dataset_inst import ProteinDataset, custom_collate_fn #change
from model import ProteinFunctionLLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NameEncoder(nn.Module):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", device="cuda"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.cache = {}  # 简单内存缓存

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        # 命中缓存直接取；未命中批量编码
        outs = [None] * len(texts)
        miss_idx, miss_texts = [], []
        for i, t in enumerate(texts):
            if t in self.cache:
                outs[i] = self.cache[t]
            else:
                miss_idx.append(i); miss_texts.append(t if t is not None else "")
        if miss_texts:
            batch = self.tok(miss_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.enc(**batch).last_hidden_state                 # [B,L,D_name]
            mask = batch["attention_mask"].unsqueeze(-1)            # [B,L,1]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)# [B,D_name]
            pooled = pooled.unsqueeze(1)                            # [B,1,D_name] 与 H_p_pooled 对齐
            for j, i in enumerate(miss_idx):
                self.cache[texts[i]] = pooled[j].detach().to("cpu") # 缓存在CPU，省显存
                outs[i] = self.cache[texts[i]]
        return torch.stack(outs, dim=0).to(self.device, dtype=torch.bfloat16)             # [B,1,D_name]

def _auto_pick_projector_ckpt(save_dir: str) -> str:
    candidates = [f for f in os.listdir(save_dir) if f.startswith("projectors_") and f.endswith(".pt")]
    if not candidates:
        return ""
    if "projectors_best.pt" in candidates:
        return os.path.join(save_dir, "projectors_best.pt")
    gs_list = []
    for f in candidates:
        try:
            if f.startswith("projectors_gs") and f.endswith(".pt"):
                step = int(f[len("projectors_gs"):-3])
                gs_list.append((step, f))
        except:
            pass
    if gs_list:
        _, fname = max(gs_list, key=lambda x: x[0])
        return os.path.join(save_dir, fname)
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(save_dir, fn)), reverse=True)
    return os.path.join(save_dir, candidates[0])

def train():
    # --- 配置部分 ---
    config = {
        "data_save_path": "./data",
        "split": "train", # 基础 split 名称，实际路径会根据 stage 拼接
        "stage1_data_path": "./data/train.csv", # 25w条数据 CSV 路径

        "eval_csv_path": "./data/eval.csv",
        "eval_split": "eval",
        "save_every_steps": 1500,         
        "eval_every_steps": 1500,
        "early_stopping_patience": 3, 
        "keep_last_k": 3,

        "stage2_data_path": "./data/stage2_data.csv", # 10w条数据 CSV 路径
        "decoder_model_path": "./re_llama", 
        "struct_embed_dim": 128,   # ProteinMPNN的输出维度 
        "seq_embed_dim": 2048,     # xTrimoPGLM的输出维度 

        "batch_size": 2,
        "stage1_epochs": 2,
        "resume_from_accelerate": "",#state
        "resume_projectors_from": "./model/trained3_projectors_stage1_train_name_new_prompt/projectors_best.pt",
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,  # 梯度裁剪
        "stage2_epochs": 1, 
        "use_fp16": True,
        "learning_rate": 3e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_stage": "stage1", # "stage1" 或 "stage2"，

        "is_add_name": True,            
        "name_model_name": "allenai/scibert_scivocab_uncased",
        "name_embed_dim": 768, 
        "name_dropout_p": 0.3,         
    }
    config["save_dir"] = f"./model/trained4_8e_projectors_{config['current_stage']}_train_name_new_prompt"
    # 固定随机种子，增强可复现性
    seed = int(os.environ.get("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 初始化Accelerator，用于简化分布式训练和混合精度
    accelerator = Accelerator(
        mixed_precision="fp16" if config["use_fp16"] else "no",
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )

    name_encoder = None
    if config.get("is_add_name", False):
        name_encoder = NameEncoder(config["name_model_name"], device=config["device"])

    
    # 加载分词器
    tokenizer = LlamaTokenizer.from_pretrained(config["decoder_model_path"],bos_token='<s>',eos_token='</s>',unk_token="<unk>")
    if tokenizer.pad_token is None:
        print("tokenizer.pad_token is None")
        tokenizer.pad_token = tokenizer.unk_token 
        tokenizer.pad_token_id = tokenizer.unk_token_id  

    
    tokenizer.padding_side = 'left'


    # --- 修改点 6: 确定并设置占位符token ---
    placeholder_token = "<protein>" # 与 dataset.py 和 Mol-Instructions 中使用的占位符一致
    # 检查占位符token是否在tokenizer中
    tokenizer.add_special_tokens({'additional_special_tokens': [placeholder_token]})
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    if placeholder_token_id == tokenizer.unk_token_id:
        print(f"Warning: Placeholder token '{placeholder_token}' not found in tokenizer. You might need to add it as a special token.")
    print(f"Using placeholder token '{placeholder_token}' with ID: {placeholder_token_id}")
    

    # --- 根据阶段设置数据加载器 ---
    if config["current_stage"] == "stage1":
        dataset = ProteinDataset(
            csv_path=config["stage1_data_path"],
            processed_dir=config["data_save_path"],
            split=config["split"] # 假设数据文件结构是 data/{split}/...
        )
        num_epochs = config["stage1_epochs"]
        print(f"Starting Stage 1 training with {len(dataset)} samples for {num_epochs} epochs.")
    elif config["current_stage"] == "stage2":
        dataset = ProteinDataset(
            csv_path=config["stage2_data_path"],
            processed_dir=config["data_save_path"],
            split=config["split"]
        )
        num_epochs = config["stage2_epochs"]
        print(f"Starting Stage 2 training with {len(dataset)} samples for {num_epochs} epochs.")
    else:
        raise ValueError(f"Invalid current_stage in config: {config['current_stage']}")

    eval_dataset  = ProteinDataset(csv_path=config["eval_csv_path"],
                          processed_dir=config["data_save_path"],
                          split=config["eval_split"])

    collate_fn_with_tokenizer = partial(custom_collate_fn, 
                                        tokenizer=tokenizer,
                                        name_encoder=name_encoder,
                                        use_name=config.get("is_add_name", False),
                                        name_dropout_p=config.get("name_dropout_p", 0.0),
                                        )

    collate_eval = partial(custom_collate_fn, 
                                        tokenizer=tokenizer,
                                        name_encoder=name_encoder,
                                        use_name=config.get("is_add_name", False),
                                        name_dropout_p=0.0
                                        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_eval
    )

    
    
    model = ProteinFunctionLLM(config)
    model.decoder.resize_token_embeddings(len(tokenizer))
    # --- 设置占位符token ID ---
    model.placeholder_token_id = placeholder_token_id

    # --- 根据阶段设置可训练参数 ---
    model.set_trainable_parameters(config["current_stage"])

    # 定义优化器 (只优化可训练的参数)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config["learning_rate"],
        weight_decay=0.01  # 添加权重衰减
    )
    bleu_metric  = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bert_metric  = evaluate.load("bertscore")
    @torch.no_grad()
    def text_metrics_once(model, loader, tokenizer, max_batches=8):
        model.eval()
        preds, refs = [], []

        gen_kwargs = dict(
            do_sample=False, num_beams=5, max_new_tokens=150,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )

        device = next(model.parameters()).device
        dec_dtype = next(model.decoder.parameters()).dtype

        batches = 0
        for batch in loader:
            if batch is None:
                continue
            struct = batch["struct_embeds"].to(device)
            seq    = batch["seq_embeds"].to(device)
            inst   = {k: v.to(device) for k,v in batch["instruction_tokens"].items()}

            proj_s = model.structure_projector(struct)
            proj_q = model.sequence_projector(seq)
            Hp     = (proj_s + proj_q).mean(dim=1, keepdim=True).to(dtype=dec_dtype)

            if getattr(model, "is_add_name", False) and ("name_embeds" in batch):
                name_w_dtype = next(model.name_projector.parameters()).dtype
                gate_w_dtype = next(model.gate_fc.parameters()).dtype
                hname = batch["name_embeds"].to(device).to(dtype=name_w_dtype)
                hname = model.name_projector(hname)               # at name_w_dtype
                h_for_gate = hname.to(dtype=gate_w_dtype)
                gate  = torch.sigmoid(model.gate_fc(h_for_gate))  # at gate_w_dtype
                alpha = float(getattr(model, "name_alpha", 1.0))

                hname = hname.to(dtype=dec_dtype)
                gate  = gate.to(dtype=dec_dtype)
                Hp    = (Hp + alpha * gate * hname).to(dtype=dec_dtype)

            input_ids     = inst["input_ids"]
            inputs_embeds = model.decoder.get_input_embeddings()(input_ids).to(dtype=dec_dtype)
            ph_hits = (input_ids == model.placeholder_token_id).sum().item()
            ph_total = input_ids.numel()
            # 只在主进程偶尔打印，避免刷屏
            if batches == 0:  # 只打印第一个 batch 也行，或改成每N个batch打印
                print(f"[Sanity] <protein> hits in batch: {ph_hits}/{ph_total}")
            mask = (input_ids == model.placeholder_token_id).unsqueeze(-1)
            inputs_embeds = torch.where(mask, Hp.expand_as(inputs_embeds), inputs_embeds)

            gen_ids   = model.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=inst["attention_mask"], **gen_kwargs)
            pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            ref_text  = tokenizer.batch_decode(batch["answer_tokens"]["input_ids"], skip_special_tokens=True)

            preds.extend(pred_text); refs.extend(ref_text)
            batches += 1
            if batches >= max_batches:
                break

        res_bleu = bleu_metric.compute(predictions=preds, references=refs)
        res_rouge = rouge_metric.compute(predictions=preds, references=refs)
        res_bert  = bert_metric.compute(predictions=preds, references=refs,
                                        model_type="dmis-lab/biobert-large-cased-v1.1",
                                        num_layers=24, lang="en")
        f1_mean = float(sum(res_bert["f1"]) / max(1, len(res_bert["f1"])))
        model.train()
        return {"bleu": float(res_bleu.get("bleu", 0.0)),
                "rougeL": float(res_rouge.get("rougeL", 0.0)),
                "bert_f1": f1_mean,
                "n_pairs": len(preds)}
    # 定义训练步数
    total_steps = (num_epochs * len(dataloader)) // config["gradient_accumulation_steps"]

    # 定义学习率调度器
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        # warmup 与训练步数以“微步数”（每个 batch 调一次 step）为基准
        num_warmup_steps=int(0.03 * (num_epochs * len(dataloader))),
        num_training_steps=num_epochs * len(dataloader),
    )

    resume_pj = config.get("resume_projectors_from", "")
    if not resume_pj:
        # 若用户未指定，自动从save_dir里挑一个
        if os.path.isdir(config["save_dir"]):
            resume_pj = _auto_pick_projector_ckpt(config["save_dir"])

    if resume_pj and os.path.isfile(resume_pj):
        ckpt = torch.load(resume_pj, map_location="cpu")
        missing = []
        try:
            model.structure_projector.load_state_dict(ckpt["structure_projector"])
        except Exception as e:
            missing.append(f"structure_projector: {e}")
        try:
            model.sequence_projector.load_state_dict(ckpt["sequence_projector"])
        except Exception as e:
            missing.append(f"sequence_projector: {e}")
        if "name_projector" in ckpt and hasattr(model, "name_projector"):
            model.name_projector.load_state_dict(ckpt["name_projector"])
        if "gate_fc" in ckpt and hasattr(model, "gate_fc"):
            model.gate_fc.load_state_dict(ckpt["gate_fc"])
        if missing:
            print("[Warm-Start] Partial load:", "; ".join(missing))
        print(f"[Warm-Start] Loaded projectors from: {resume_pj}")
    else:
        print("[Warm-Start] No projector ckpt found; training from current init.")

    best_val = float("inf")
    last_ckpts = []
    global_step = 0
    epochs_without_improve = 0
    last_eval_val_loss = None
    
    # 使用Accelerator准备所有组件
    model, optimizer, dataloader,eval_dataloader,lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader,eval_dataloader,lr_scheduler
    )
    # === 新增：如需断点续训，加载保存的训练状态 ===
    resume_dir = config.get("resume_from_accelerate", "")
    if resume_dir:
        if os.path.isfile(resume_dir):
            print(f"[RESUME] '{resume_dir}' 是 .pt 文件。accelerate.load_state 需要目录 (state_gsXXXX/)。"
                f"请改填 state_* 目录或把 resume_from_accelerate 置空。")
        elif os.path.isdir(resume_dir):
            accelerator.load_state(resume_dir)   # 恢复模型/优化器/调度器/混精/RNG
            try:
                with open(os.path.join(resume_dir, "meta.json"), "r") as f:
                    meta = json.load(f)
                best_val = float(meta.get("best_val", best_val))
                global_step = int(meta.get("global_step", global_step))
                last_eval_val_loss = meta.get("val_loss_last_eval", last_eval_val_loss)
                print(f"[RESUME] Loaded accelerate state from: {resume_dir} "
                    f"(global_step={global_step}, best_val={best_val:.4f})")
            except Exception as e:
                print(f"[RESUME] meta.json not found or unreadable: {e}")
        else:
            print(f"[RESUME] 状态目录不存在：{resume_dir}")

     # ============ 验证/保存工具函数 ============
    @torch.no_grad()
    def validate_once() -> float:
        model.eval()
        total, n = 0.0, 0
        for batch in eval_dataloader:
            if batch is None:
                continue
            out = model(batch)
            total += float(out.loss)
            n += 1
        dev = next(model.parameters()).device  
        total_all = accelerator.gather_for_metrics(torch.tensor([total], device=dev)).sum().item()
        n_all     = accelerator.gather_for_metrics(torch.tensor([n],     device=dev)).sum().item()
        model.train()
        return (total_all / max(n_all, 1)) if n_all > 0 else float("inf")

    def save_projectors(tag: str) -> str:
        unwrapped = accelerator.unwrap_model(model)
        sd = {
            'structure_projector': unwrapped.structure_projector.state_dict(),
            'sequence_projector':  unwrapped.sequence_projector.state_dict(),
            'config': config,
            'stage':  config['current_stage']
        }
        if getattr(unwrapped, "is_add_name", False) and hasattr(unwrapped, "name_projector"):
            sd["name_projector"] = unwrapped.name_projector.state_dict()
            sd["gate_fc"] = unwrapped.gate_fc.state_dict()
        os.makedirs(config["save_dir"], exist_ok=True)
        path = os.path.join(config["save_dir"], f"projectors_{tag}.pt")
        torch.save(sd, path)
        return path



    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        step_count = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            disable=not accelerator.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            if batch is None: 
                continue

            try:
                with accelerator.accumulate(model):
                    # 前向传播
                    outputs = model(batch)
                    loss = outputs.loss
                    
                    # 反向传播
                    accelerator.backward(loss)
                    
                    # 梯度裁剪
                    if accelerator.sync_gradients:
                        if config["max_grad_norm"] > 0:
                            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    step_count += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss/step_count:.4f}"
                    })
                    
                if accelerator.sync_gradients:
                    global_step += 1

                    # 按步评估
                    if (global_step % config["eval_every_steps"] == 0):
                        val_loss = validate_once()
                        last_eval_val_loss = float(val_loss)
                        accelerator.print(f"[Eval] step {global_step}: val_loss={val_loss:.4f}")
                        # 保存 best
                        if accelerator.is_main_process:
                            tm = text_metrics_once(model, eval_dataloader, tokenizer, max_batches=50)
                            accelerator.print(f"[TextEval@step {global_step}] "
                                            f"BLEU={tm['bleu']:.4f} | ROUGE-L={tm['rougeL']:.4f} | BERT-F1={tm['bert_f1']:.4f} "
                                            f"(N={tm['n_pairs']})")

                            # —— 排名键：优先 ROUGE-L，其次 BERT-F1，最后用较小的 val_loss 兜底 —— #
                            rank_key = (round(tm['rougeL'], 6), round(tm['bert_f1'], 6), -round(val_loss, 6))
                            if rank_key > getattr(train, "_best_text_rank_key", (-1.0, -1.0, float("-inf"))):
                                train._best_text_rank_key = rank_key
                                save_projectors("best_text")
                                accelerator.print(f"[Checkpoint] New best_text at step {global_step} "
                          f"(ROUGE-L={tm['rougeL']:.4f}, BERT-F1={tm['bert_f1']:.4f}, val_loss={val_loss:.4f})")
                        if val_loss < best_val:
                            best_val = val_loss
                            if accelerator.is_main_process:
                                save_projectors("best")
                            accelerator.print(f"[Checkpoint] New best at step {global_step} (val_loss={val_loss:.4f})")

                    # 按步保存“最近K个”
                    if (global_step % config["save_every_steps"] == 0) and accelerator.is_main_process:
                        ckpt_path = save_projectors(f"gs{global_step}")
                        last_ckpts.append(ckpt_path)
                        acc_state_dir = os.path.join(config["save_dir"], f"state_gs{global_step}")  # ← 新增
                        os.makedirs(acc_state_dir, exist_ok=True)
                        accelerator.save_state(acc_state_dir)

                        try:
                            curr_lr = float(lr_scheduler.get_last_lr()[0])
                        except Exception:
                            curr_lr = float(optimizer.param_groups[0]["lr"])

                        stats = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "stage": config["current_stage"],
                            "epoch": epoch + 1,
                            "step_in_epoch": step,
                            "global_step": global_step,
                            "loss_last": float(loss.item()),
                            "avg_loss_epoch_so_far": float(total_loss / max(step_count, 1)),
                            "val_loss_last_eval": None if last_eval_val_loss is None else float(last_eval_val_loss),
                            "best_val": float(best_val),
                            "lr": curr_lr,
                            "batch_size": config["batch_size"],
                            "grad_accum_steps": config["gradient_accumulation_steps"],
                        }

                        with open(os.path.join(acc_state_dir, "meta.json"), "w") as f:
                            json.dump(stats, f, indent=2)

                        while len(last_ckpts) > config["keep_last_k"]:
                            old = last_ckpts.pop(0)
                            try:
                                os.remove(old)
                            except:
                                pass

                            base = os.path.basename(old)  # e.g., projectors_gs2000.pt
                            if base.startswith("projectors_gs") and base.endswith(".pt"):
                                step = base[len("projectors_gs"):-3]
                                state_dir = os.path.join(config["save_dir"], f"state_gs{step}")
                                if os.path.isdir(state_dir):
                                    import shutil
                                    try:
                                        shutil.rmtree(state_dir)
                                    except:
                                        pass

                    # 定期清理显存
                    if step % 200 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at step {step}: {e}")
                raise e

            #长度不齐触发频率
            except Exception as e:
                print(f"Error at step {step}: {e}")
                if "lengths do not match" in str(e):
                    with open("length_mismatch_log.txt", "a") as f:
                        f.write(f"Epoch {epoch+1}, Step {step}: {e}\n")
                continue

        avg_loss = total_loss / max(step_count, 1)
        accelerator.print(f"Stage {config['current_stage']}, Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # 每个 epoch 末再跑一次验证 + 早停逻辑
        val_loss = validate_once()
        accelerator.print(f"[Val@EpochEnd] Epoch {epoch+1}: val_loss={val_loss:.4f}")
        if accelerator.is_main_process:
            tm = text_metrics_once(model, eval_dataloader, tokenizer, max_batches=50)
            accelerator.print(f"[TextEval@step {global_step}] "
                            f"BLEU={tm['bleu']:.4f} | ROUGE-L={tm['rougeL']:.4f} | BERT-F1={tm['bert_f1']:.4f} "
                            f"(N={tm['n_pairs']})")

            # —— 排名键：优先 ROUGE-L，其次 BERT-F1，最后用较小的 val_loss 兜底 —— #
            rank_key = (round(tm['rougeL'], 6), round(tm['bert_f1'], 6), -round(val_loss, 6))
            if rank_key > getattr(train, "_best_text_rank_key", (-1.0, -1.0, float("-inf"))):
                train._best_text_rank_key = rank_key
                save_projectors("best_text")
                accelerator.print(f"[Checkpoint] New best_text at step {global_step} "
            f"(ROUGE-L={tm['rougeL']:.4f}, BERT-F1={tm['bert_f1']:.4f}, val_loss={val_loss:.4f})")
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            if accelerator.is_main_process:
                save_projectors("best")
            accelerator.print(f"[Checkpoint] New best at epoch {epoch+1} (val_loss={val_loss:.4f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.get("early_stopping_patience", 0):
                accelerator.print(f"[EarlyStopping] No improvement for {epochs_without_improve} epochs. Stop training.")
                break

        # 每个 epoch 保存一份 last（可选）
        if accelerator.is_main_process:
            save_projectors(f"epoch{epoch+1}_last")
            acc_state_dir = os.path.join(config["save_dir"], f"state_epoch{epoch+1}_last")
            os.makedirs(acc_state_dir, exist_ok=True) 
            accelerator.save_state(acc_state_dir)

            try:
                curr_lr = float(lr_scheduler.get_last_lr()[0])
            except Exception:
                curr_lr = float(optimizer.param_groups[0]["lr"])

            stats = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "stage": config["current_stage"],
                "epoch": epoch + 1,
                "step_in_epoch": step,              # 末步
                "global_step": global_step,
                "loss_last": float(loss.item()) if 'loss' in locals() else None,
                "avg_loss_epoch_so_far": float(total_loss / max(step_count, 1)),
                "val_loss_last_eval": None if last_eval_val_loss is None else float(last_eval_val_loss),
                "best_val": float(best_val),
                "lr": curr_lr,
                "batch_size": config["batch_size"],
                "grad_accum_steps": config["gradient_accumulation_steps"],
            }
            with open(os.path.join(acc_state_dir, "meta.json"), "w") as f:
                json.dump(stats, f, indent=2)

        # 每个epoch结束后清理显存
        torch.cuda.empty_cache()
        gc.collect()

    accelerator.print("Training completed.")
  


if __name__ == "__main__":
    # 在开始训练前清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Training completed or terminated.")