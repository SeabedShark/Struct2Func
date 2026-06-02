import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from functools import partial
from tqdm import tqdm
import shutil
    
from dataset import ProteinDataset, custom_collate_fn
# temperature=0.9, top_p=0.9, top_k=8, num_beams=1, repetition_penalty=1.2, max_new_tokens=1024
class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 使用一个简单的线性层进行投影
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

class ProteinFunctionLLM(nn.Module):
    """
    集成的多模态模型。
    它包含两个MLP投影器和LLaMA解码器。
    """
    def __init__(self, config):
        """
        Args:
            config: 包含模型配置的字典。
        """
        super().__init__()
        self.config = config
        self.device = config.get("device", "cuda")
        self.placeholder_token_id = None
        
        # 加载LLM
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                config["decoder_model_path"],
                torch_dtype=torch.bfloat16 if self.device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32,
                # low_cpu_mem_usage=True, # 如果内存紧张可以启用
                # device_map="auto" # 如果模型太大，可以自动分配到多卡 (与 Accelerator 可能冲突)
            )
            self.decoder.gradient_checkpointing_enable()
            self.decoder.config.use_cache = False
            print(f"LLM {config['decoder_model_path']} loaded successfully.")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise e
        decoder_hidden_dim = self.decoder.config.hidden_size

        self.is_add_name = bool(config.get("is_add_name", False))
        if self.is_add_name:
            self.name_embed_dim = int(config.get("name_embed_dim", 768))
            self.name_projector = MLP(self.name_embed_dim, decoder_hidden_dim)
            self.gate_fc = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        print(f"LLM Hidden Dim: {decoder_hidden_dim}")

        # 初始化两个投影器，分别用于结构嵌入和序列嵌入
        self.structure_projector = MLP(
            input_dim=config["struct_embed_dim"],
            output_dim=decoder_hidden_dim,
        )
        self.sequence_projector = MLP(
            input_dim=config["seq_embed_dim"],
            output_dim=decoder_hidden_dim,
        )
        decoder_dtype = next(self.decoder.parameters()).dtype

        # Convert all projectors to match decoder dtype
        self.structure_projector = self.structure_projector.to(dtype=decoder_dtype)
        self.sequence_projector = self.sequence_projector.to(dtype=decoder_dtype)
        if self.is_add_name:
            self.name_projector = self.name_projector.to(dtype=decoder_dtype)
            self.gate_fc = self.gate_fc.to(dtype=decoder_dtype)

        print(f"[Model Initialization] Projectors dtype aligned to decoder: {decoder_dtype}")

    def set_trainable_parameters(self, stage):
        """
        根据训练阶段设置可训练的参数。
        Args:
            stage (str): "stage1" 或 "stage2"
        """
        # 默认全部冻结
        for param in self.parameters():
            param.requires_grad = False

        if stage in ["stage1","stage2"]:
            for p in self.structure_projector.parameters(): p.requires_grad = True
            for p in self.sequence_projector.parameters():  p.requires_grad = True
            if self.is_add_name:
                for p in self.name_projector.parameters(): p.requires_grad = True
                for p in self.gate_fc.parameters():       p.requires_grad = True

        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params} / {total_params}")

    def forward(self, batch):
        """
        模型的前向传播逻辑。
        Args:
            batch: 来自 DataLoader 的批次数据字典。
        Returns:
            outputs: LLM 的输出 (包含 loss, logits 等)。
        """
        # --- 数据提取 ---
        struct_embeds = batch['struct_embeds'].to(self.device) # [B, L_struct, D_struct]
        seq_embeds = batch['seq_embeds'].to(self.device)       # [B, L_seq, D_seq]
        # struct_masks = batch['struct_masks'].to(self.device)   # [B, L_struct] (可选)
        # seq_masks = batch['seq_masks'].to(self.device)         # [B, L_seq] (可选)
        instruction_tokens = {k: v.to(self.device) for k, v in batch['instruction_tokens'].items()}
        answer_tokens = {k: v.to(self.device) for k, v in batch['answer_tokens'].items()}

        batch_size = struct_embeds.size(0)

        # --- 1. 投影 (Projection) ---
        
        projected_struct = self.structure_projector(struct_embeds) # [B, L_struct, D_llm]
        projected_seq = self.sequence_projector(seq_embeds)         # [B, L_seq, D_llm]
        
        target_dtype = next(self.decoder.parameters()).dtype
        projected_struct = projected_struct.to(dtype=target_dtype)
        projected_seq    = projected_seq.to(dtype=target_dtype)

        # --- 2. 融合 (Fusion) ---
        # 假设 L_struct == L_seq (由 collate_fn 保证)
        if projected_struct.size(1) != projected_seq.size(1):
             raise ValueError(f"Projected struct ({projected_struct.shape}) and seq ({projected_seq.shape}) lengths do not match for element-wise addition.")
        fused_protein_embeds = projected_struct + projected_seq # [B, L, D_llm]

        # --- 3. 池化为软提示 (Pooling to Soft Prompt) ---
        # 遵循 EvoLLaMA，使用平均池化得到单个软提示向量
        H_p_pooled = fused_protein_embeds.mean(dim=1, keepdim=True) # [B, 1, D_llm]
        # 如果需要多个软提示 token，可以调整池化策略或直接使用 H_p_pooled.repeat(1, N, 1)
        if self.is_add_name and ("name_embeds" in batch):

            name_w_dtype = next(self.name_projector.parameters()).dtype
            gate_w_dtype = next(self.gate_fc.parameters()).dtype
            dec_dtype    = next(self.decoder.parameters()).dtype

            h_name = batch["name_embeds"].to(self.device).to(dtype=name_w_dtype)
            h_name_proj = self.name_projector(h_name)              # at name_w_dtype
            h_for_gate  = h_name_proj.to(dtype=gate_w_dtype)
            gate        = torch.sigmoid(self.gate_fc(h_for_gate))  # at gate_w_dtype

            h_name_proj = h_name_proj.to(dtype=dec_dtype)
            gate        = gate.to(dtype=dec_dtype)
            H_p_pooled  = H_p_pooled + gate * h_name_proj      # Gated Add

        # --- 4. 获取指令 token 的嵌入 ---
        instruction_input_ids = instruction_tokens['input_ids'] # [B, L_inst]
        instruction_embeds = self.decoder.get_input_embeddings()(instruction_input_ids) # [B, L_inst, D_llm]

        # --- 5. 软提示替换 (Soft Prompt Replacement) ---
       
        if self.placeholder_token_id is None:
            raise ValueError("placeholder_token_id is not set in the model. Please set it before training.")
        # 查找占位符 token 的位置
        placeholder_positions = (instruction_input_ids == self.placeholder_token_id) # [B, L_inst] Bool Tensor
        # print(f"DEBUG [model.py]: placeholder_positions.any() = {placeholder_positions.any()}")
        if not placeholder_positions.any():
            print(f"Warning: Placeholder token '{self.placeholder_token}' (ID: {self.placeholder_token_id}) not found in a batch. Skipping replacement for this batch.")

            final_inputs_embeds = instruction_embeds
            final_attention_mask = instruction_tokens.get('attention_mask', torch.ones_like(instruction_input_ids))
        else:
            # 创建一个副本用于修改
            modified_embeds = instruction_embeds.clone() # [B, L_inst, D_llm]
            # print(f"DEBUG: batch_size={batch_size}")
            # print(f"DEBUG: instruction_input_ids.shape={instruction_input_ids.shape}")
            # print(f"DEBUG: instruction_embeds.shape={instruction_embeds.shape}")
            # print(f"DEBUG: modified_embeds.shape={modified_embeds.shape}")
            # print(f"DEBUG: placeholder_positions.shape={placeholder_positions.shape}")
            # print(f"DEBUG: H_p_pooled.shape={H_p_pooled.shape}")
            
            # 将占位符位置的嵌入替换为软提示 H_p_pooled
 
            for b in range(batch_size):
                pos_indices = torch.where(placeholder_positions[b])[0] # where返回所有 True 的索引
                for pos_idx in pos_indices:
                    modified_embeds[b, pos_idx, :] = H_p_pooled[b, 0, :] # 替换为软提示
            
            final_inputs_embeds = modified_embeds
            final_attention_mask = instruction_tokens.get('attention_mask', torch.ones_like(instruction_input_ids))

        # --- 6. 拼接答案 (用于训练) ---
        labels = None
        if answer_tokens is not None and 'input_ids' in answer_tokens:
            answer_input_ids = answer_tokens['input_ids'] # [B, L_ans]
            answer_embeds = self.decoder.get_input_embeddings()(answer_input_ids) # [B, L_ans, D_llm]
            
            # 拼接嵌入和注意力掩码
            final_inputs_embeds = torch.cat([final_inputs_embeds, answer_embeds], dim=1) # [B, L_inst + L_ans, D_llm]
            
            answer_attention_mask = answer_tokens.get('attention_mask', torch.ones_like(answer_input_ids))
            final_attention_mask = torch.cat([final_attention_mask, answer_attention_mask], dim=1) # [B, L_inst + L_ans]

            # --- 7. 构建 Labels (用于计算 loss) ---
            inst_len = instruction_input_ids.size(1)
            ans_len = answer_input_ids.size(1)
            total_len = inst_len + ans_len
            
            # 初始化 labels 为 -100 (ignored by CrossEntropyLoss)
            labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=self.device)
            # 答案部分的 labels 是 answer_input_ids
            labels[:, inst_len:] = answer_input_ids # [B, L_ans]
            # 指令部分保持 -100，模型不会计算这部分的 loss

        # --- 8. 输入到 LLM ---
        outputs = self.decoder(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            labels=labels, # 用于训练时计算 loss
            output_hidden_states=False, # 根据需要设置
            output_attentions=False,
            return_dict=True
        )

        return outputs # 包含 loss (如果 labels 提供), logits 等