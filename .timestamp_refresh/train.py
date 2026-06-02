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


def train():
    # --- 配置部分 ---
    config = {
        "data_save_path": "./data",
        "split": "test", # 基础 split 名称，实际路径会根据 stage 拼接
        "stage1_data_path": "./data/test.csv", # 25w条数据 CSV 路径
        "stage2_data_path": "./data/stage2_data.csv", # 10w条数据 CSV 路径
        "decoder_model_path": "./re_llama", 
        "struct_embed_dim": 128,   # ProteinMPNN的输出维度 
        "seq_embed_dim": 2048,     # xTrimoPGLM的输出维度 
        "batch_size": 2,
        "stage1_epochs": 5,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,  # 梯度裁剪
        "stage2_epochs": 1, 
        "use_fp16": True,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_stage": "stage1", # "stage1" 或 "stage2"，

        "is_add_name": True,            
        "name_model_name": "allenai/scibert_scivocab_uncased",
        "name_embed_dim": 768, 
        "name_dropout_p": 0.3,         
    }
    config["save_dir"] = f"./model/trained_projectors_{config['current_stage']}_test_name_new_prompt"
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

    collate_fn_with_tokenizer = partial(custom_collate_fn, 
                                        tokenizer=tokenizer,
                                        name_encoder=name_encoder,
                                        use_name=config.get("is_add_name", False),
                                        name_dropout_p=config.get("name_dropout_p", 0.0),
                                        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer
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

    total_steps = (num_epochs * len(dataloader)) // config["gradient_accumulation_steps"]

    # 定义学习率调度器
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=num_epochs * len(dataloader),
    )

    # 使用Accelerator准备所有组件
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

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
                    
                    # 定期清理显存
                    if step % 50 == 0:
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


        # 每个epoch结束后清理显存
        torch.cuda.empty_cache()
        gc.collect()

    # 保存训练好的模型（只保存可训练的部分，即投影器）
    accelerator.wait_for_everyone()
    # 使用 accelerator.unwrap_model 获取原始模型
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 创建保存目录
    output_dir = config["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存投影器的状态字典
    save_dict = {
        'structure_projector': unwrapped_model.structure_projector.state_dict(),
        'sequence_projector': unwrapped_model.sequence_projector.state_dict(),
        'config': config,
        'stage': config['current_stage']
    }
    # 新增：若启用了名字分支，同时保存 name_projector & gate_fc
    if getattr(unwrapped_model, "is_add_name", False) and hasattr(unwrapped_model, "name_projector"):
        save_dict["name_projector"] = unwrapped_model.name_projector.state_dict()
        save_dict["gate_fc"] = unwrapped_model.gate_fc.state_dict()
        
    save_path = os.path.join(output_dir, f"projectors_and_config_{config['current_stage']}.pt")
    torch.save(save_dict, save_path)
    
    accelerator.print(f"Stage {config['current_stage']} training complete. Weights saved to {save_path}")


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