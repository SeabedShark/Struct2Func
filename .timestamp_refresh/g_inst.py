
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import torch.nn as nn
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,AutoModel
import shutil

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
import random
random.seed(42)

# --- 直接从 preprocess_data.py 导入 ---
try:
    from preprocess_data import ProteinMPNNEmbedder, XtrimoPGLMEmbedder, download_alphafold_structure
    print("Successfully imported embedders from preprocess_data.py")
except ImportError as e:
    print(f"Error importing from preprocess_data.py: {e}")
    print("Please ensure preprocess_data.py is in the same directory or in the Python path.")
    sys.exit(1)

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

# --- 1. 复用模型架构 (与 model.py / model.txt 完全一致) ---
class MLP(nn.Module):
    """
    MLP 投影器。与 model.py 中的定义完全一致。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # --- 关键：使用 self.layers，与 model.py 一致 ---
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # --- 关键：调用 self.layers，与 model.py 一致 ---
        return self.layers(x)

class ProteinFunctionLLM(nn.Module):
    """
    集成的多模态模型。与 model.py 中的定义核心部分一致。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cuda")
        self.placeholder_token_id = None
        self.placeholder_token = None

        # 加载LLM (推理时加载)
        try:
            print(f"Loading LLM from {config['decoder_model_path']}...")
            self.decoder = AutoModelForCausalLM.from_pretrained(
                config["decoder_model_path"],
                torch_dtype=torch.bfloat16 if self.device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32,
            ).to(self.device)
            print(f"LLM {config['decoder_model_path']} loaded successfully.")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise e

        decoder_hidden_dim = self.decoder.config.hidden_size
        print(f"LLM Hidden Dim: {decoder_hidden_dim}")

        # 获取LLM的数据类型，确保投影器与LLM类型一致
        self.target_dtype = next(self.decoder.parameters()).dtype

        print(f"Target dtype for projectors: {self.target_dtype}")
        self.is_add_name = bool(config.get("is_add_name", False))
        if self.is_add_name:
            self.name_embed_dim = int(config.get("name_embed_dim", 768))
            self.name_projector = MLP(self.name_embed_dim, decoder_hidden_dim).to(self.device).to(dtype=self.target_dtype)
            self.gate_fc = nn.Linear(decoder_hidden_dim, decoder_hidden_dim).to(self.device).to(dtype=self.target_dtype)

        
        self.structure_projector = MLP(
            input_dim=config["struct_embed_dim"],
            output_dim=decoder_hidden_dim,
        ).to(self.device).to(dtype=self.target_dtype)  # 确保数据类型一致
        
        self.sequence_projector = MLP(
            input_dim=config["seq_embed_dim"],
            output_dim=decoder_hidden_dim,
        ).to(self.device).to(dtype=self.target_dtype)  # 确保数据类型一致

    def load_projector_weights(self, weights_path):
        """从文件加载训练好的投影器权重。"""
        print(f"Loading trained projector weights from {weights_path}...")
        try:
            # --- 修正：添加 weights_only=True ---
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            
            # 加载权重到投影器
            self.structure_projector.load_state_dict(checkpoint['structure_projector'])
            self.sequence_projector.load_state_dict(checkpoint['sequence_projector'])
            if "name_projector" in checkpoint and getattr(self, "is_add_name", False):
                self.name_projector.load_state_dict(checkpoint["name_projector"])
                self.gate_fc.load_state_dict(checkpoint["gate_fc"])
                self.name_projector = self.name_projector.to(dtype=self.target_dtype)
                self.gate_fc = self.gate_fc.to(dtype=self.target_dtype)
            
            # 重要：加载权重后，确保投影器与LLM数据类型一致
            self.structure_projector = self.structure_projector.to(dtype=self.target_dtype)
            self.sequence_projector = self.sequence_projector.to(dtype=self.target_dtype)

            
            print(f"Projector weights loaded and converted to {self.target_dtype}.")
        except Exception as e:
            print(f"Failed to load projector weights: {e}")
            raise e
            
    def forward(self, struct_embeds, seq_embeds,name_pooled: torch.Tensor = None):
        """
        模型的前向传播逻辑，用于生成软提示。
        Args:
            struct_embeds (torch.Tensor): 结构嵌入 [1, L, D_struct]
            seq_embeds (torch.Tensor): 序列嵌入 [1, L, D_seq]
        Returns:
            torch.Tensor: 池化后的软提示 [1, D_llm]
        """
        # --- 1. 投影 ---
        projected_struct = self.structure_projector(struct_embeds) # [1, L, D_llm]
        projected_seq = self.sequence_projector(seq_embeds)       # [1, L, D_llm]

        # --- 2. 融合 ---
        if projected_struct.size(1) != projected_seq.size(1):
             raise ValueError(f"Projected struct ({projected_struct.shape}) and seq ({projected_seq.shape}) lengths do not match for element-wise addition.")
        fused_protein_embeds = projected_struct + projected_seq # [1, L, D_llm]

        # --- 3. 池化为软提示 ---
        H_p_pooled = fused_protein_embeds.mean(dim=1, keepdim=True) # [1, D_llm] 无keepdim
        if self.is_add_name and (name_pooled is not None):
        # name_pooled: [B,1,D_name]
            h_name_proj = self.name_projector(name_pooled.to(self.device))
            h_name_proj = h_name_proj.to(dtype=projected_struct.dtype)       # dtype 对齐
            gate = torch.sigmoid(self.gate_fc(h_name_proj))                  # [B,1,D]
            H_p_pooled = H_p_pooled + gate * h_name_proj
        return H_p_pooled

# --- 2. 核心预测器类 ---
class ProteinFunctionPredictor:
    def __init__(self, config):

        self.config = config
        self.device = config["device"]
        
        print("--- Initializing Components ---")
        
        # 1. 加载分词器
        print("1. Loading tokenizer...")
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                config["decoder_model_path"],
                bos_token='<s>',
                eos_token='</s>',
                unk_token="<unk>"
            )
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                print("tokenizer.pad_token is None")
                self.tokenizer.pad_token = self.tokenizer.unk_token 
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id  
            print("  -> Tokenizer loaded.")
        except Exception as e:
            print(f"  -> Failed to load tokenizer: {e}")
            raise e
        
        print(self.tokenizer.bos_token, self.tokenizer.bos_token_id)  # <s> 1
        print(self.tokenizer.eos_token, self.tokenizer.eos_token_id)  # </s> 2
        print(self.tokenizer.pad_token, self.tokenizer.pad_token_id)  # <pad> 0
        print(self.tokenizer.unk_token, self.tokenizer.unk_token_id)  # <unk> 0 或其他
        print(self.tokenizer.add_bos_token, self.tokenizer.add_eos_token)  # True
        
        # 2. 加载训练好的投影器
        print("2. Loading trained MLP projectors...")
        try:
            self.model = ProteinFunctionLLM(config).to(self.device)
            self.model.load_projector_weights(config["model_weights_path"])
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            # 设置模型所需的占位符 token ID
            placeholder_token = "<protein>" # 与训练和 dataset.py 一致
            self.model.placeholder_token = placeholder_token
            self.tokenizer.add_special_tokens({'additional_special_tokens': [placeholder_token]})
            
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))
            
            placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)

            if placeholder_token_id == self.tokenizer.unk_token_id:
                print(f"Warning: Placeholder token '{placeholder_token}' not found in tokenizer.")
            self.model.placeholder_token_id = placeholder_token_id
            print(f"  -> Placeholder token '{placeholder_token}' set with ID: {placeholder_token_id}")
            print("  -> Trained MLP projectors loaded.")
        except Exception as e:
            print(f"  -> Failed to load trained projectors: {e}")
            raise e

        self.name_encoder = None
        if self.config.get("is_add_name", False):
            print("2.5 Initializing NameEncoder...")
            self.name_encoder = NameEncoder(
                model_name=self.config.get("name_model_name", "allenai/scibert_scivocab_uncased"),
                device=self.device
            )
            print("  -> NameEncoder ready.")

        # 3. 初始化蛋白质嵌入提取器 (直接复用)
        print("3. Initializing protein embedders (reusing from preprocess_data.py)...")
        try:
            self.mpnn_embedder = ProteinMPNNEmbedder(
                config["protein_mpnn_weights_path"], self.device
            )
            self.mpnn_embedder.model.eval()
            for param in self.mpnn_embedder.model.parameters():
                param.requires_grad = False

            self.pglm_embedder = XtrimoPGLMEmbedder(
                config["xtrimo_model_name"], device=self.device
            )
            self.pglm_embedder.model.eval()
            for param in self.pglm_embedder.model.parameters():
                param.requires_grad = False
                
            print("  -> Protein embedders loaded successfully.")
        except Exception as e:
            print(f"  -> Failed to load protein embedders: {e}")
            raise e

    @torch.no_grad()
    def predict(self, uniprot_id, sequence,full_name: str = None):
        """对单个蛋白质进行功能预测。"""
        print(f"\n--- Starting prediction for protein: {uniprot_id} ---")
        
        # 1. 预处理：获取结构和序列嵌入
        print("Step 1/4: Downloading structure and generating embeddings...")
        pdb_dir = "./prediction_temp_pdb"
        pdb_path = download_alphafold_structure(uniprot_id, pdb_dir)
        if pdb_path is None:
            return "Error: Could not retrieve protein structure file."

        # --- 关键修正：根据 preprocess_data.py 的标准返回值进行解包 ---
        # preprocess_data.py 中 ProteinMPNNEmbedder.stru_embed 返回 (embedding, mask)
        struct_embeds_np, struct_mask_np = self.mpnn_embedder.stru_embed(pdb_path)
        # preprocess_data.py 中 XtrimoPGLMEmbedder.seq_embed 返回 (embedding, mask)
        seq_embeds_np, seq_mask_np = self.pglm_embedder.seq_embed(sequence)
        
        # 确保嵌入在正确的设备上
        # MPNN 返回 numpy array
        struct_embeds = torch.from_numpy(struct_embeds_np).to(self.device) # Shape: [L, D_struct] -> [1, L, D_struct]
        if struct_embeds.dim() == 2:
             struct_embeds = struct_embeds.unsqueeze(0)
        
        # xTrimoPGLM 返回 numpy array
        seq_embeds = torch.from_numpy(seq_embeds_np).to(self.device) # Shape: [L, D_seq] -> [1, L, D_seq]
        if seq_embeds.dim() == 2:
             seq_embeds = seq_embeds.unsqueeze(0)
            
        print(f"  -> Struct embedding shape: {struct_embeds.shape}")
        print(f"  -> Seq embedding shape: {seq_embeds.shape}")

        # 2. 投影和融合
        print("Step 2/4: Projecting and fusing protein embeddings...")
        try:
            with torch.no_grad():
                # 确保输入数据类型与投影器一致
                struct_embeds = struct_embeds.to(dtype=self.model.target_dtype)
                seq_embeds = seq_embeds.to(dtype=self.model.target_dtype)
                print(f"  -> Converted embeddings to {self.model.target_dtype}")
                name_pooled = None
                if (self.name_encoder is not None) and (full_name is not None):
                    name_pooled = self.name_encoder.encode([full_name])      # [1,1,D_name]
                
                pooled_protein_embed = self.model(struct_embeds, seq_embeds, name_pooled=name_pooled) # [1, D_llm]
            print("  -> Protein embeddings projected and fused.")
        except Exception as e:
            print(f"  -> Error during projection/fusion: {e}")
            raise e

        # 3. 构建指令并替换软提示
        print("Step 3/4: Building input with soft prompt...")
        try:
            instruction = "Given a protein represented at <protein>, describe its likely function, subcellular localization, domains/motifs, and catalytic activity."
            instruction_text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n<protein>\n\n### Response:\n"
            print(f"  -> Using instruction template: {instruction_text}")
            
            instruction_tokens = self.tokenizer(
                instruction_text,
                return_tensors='pt',
                padding=False,
                truncation=False
            )
            instruction_tokens = {k: v.to(self.device) for k, v in instruction_tokens.items()}
            instruction_input_ids = instruction_tokens['input_ids'] # [1, L_inst]
            instruction_embeds = self.model.decoder.get_input_embeddings()(instruction_input_ids) # [1, L_inst, D_llm]
            print(f"  -> Instruction tokens shape: {instruction_input_ids.shape}")

            if self.model.placeholder_token_id is None:
                raise ValueError("Model placeholder_token_id is not set.")

            placeholder_positions = (instruction_input_ids == self.model.placeholder_token_id) # [1, L_inst]
            
            if not placeholder_positions.any():
                print(f"Warning: Placeholder token '{self.placeholder_token}' (ID: {self.placeholder_token_id}) not found in a batch. Skipping replacement for this batch.")

                final_inputs_embeds = instruction_embeds
                final_attention_mask = instruction_tokens.get('attention_mask', torch.ones_like(instruction_input_ids))
            else:
                # 创建一个副本用于修改
                modified_embeds = instruction_embeds.clone() # [B, L_inst, D_llm]
                pos_indices = torch.where(placeholder_positions[0])[0] # where返回所有 True 的索引
                for pos_idx in pos_indices:
                    modified_embeds[0, pos_idx, :] = pooled_protein_embed[0, 0, :] # 替换为软提示
                
                final_inputs_embeds = modified_embeds
                final_attention_mask = instruction_tokens.get('attention_mask', torch.ones_like(instruction_input_ids))
            
            print(f"  -> Final inputs_embeds shape: {final_inputs_embeds.shape}")

        except Exception as e:
            print(f"  -> Error building input: {e}")
            raise e

        # 4. 生成文本
        print("Step 4/4: Calling LLM to generate function description...")
        try:
            with torch.no_grad():
                # logits = self.model.decoder(inputs_embeds=final_inputs_embeds).logits  
                outputs = self.model.decoder.generate(
                    inputs_embeds=final_inputs_embeds,
                    attention_mask=final_attention_mask,  # 修复：确保传递注意力掩码
                    max_new_tokens=self.config.get("max_new_tokens", 150),
                    do_sample=self.config.get("do_sample", False),
                    temperature=self.config.get("temperature", 0.7) if self.config.get("do_sample", False) else 1.0,
                    top_p=self.config.get("top_p", 0.9) if self.config.get("do_sample", False) else 1.0,
                    num_beams=self.config.get("num_beams", 1),
                    early_stopping=self.config.get("early_stopping", False),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    min_new_tokens=5,
                )
            # print("first-step max|Δlogit|:", float((logits[:, -1, :] - logits[:, -1, :]).abs().max()))
            print("  -> Text generation completed.")
            print(f"outputs.shape = {outputs.shape}")
        except Exception as e:
            print(f"  -> Error during generation: {e}")
            print("  -> Returning error message for debugging.")
            return f"[Generation failed: {e}]"

        # 5. 解码并返回结果
        try:
            # # 解码生成的文本
            # # outputs.shape is [1, L_generated]
            # input_length = final_inputs_embeds.shape[1] # 输入长度
            
            
            # if outputs.shape[1] > input_length:
            #     # generated_tokens = outputs[0, input_length:]  
            #     generated_tokens = outputs[0]  
            #     generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # else:
            #     # 如果没有生成新的token
            #     generated_text = "[No new tokens generated]"
            
            # print("--- Prediction completed ---")
            output_len = outputs.shape[1]
            input_len = instruction_input_ids.shape[1]

            # 如果 outputs 比 input 长，说明返回的是完整序列
            # if output_len > input_len:
            #     generated_tokens = outputs[0, input_len:]
            # else:
            #     # 否则 HuggingFace 只返回新 tokens
            #     generated_tokens = outputs[0]
            generated_tokens = outputs[0]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text.strip()

        except Exception as e:
            print(f"  -> Error during decoding: {e}")
            raise e
        finally:
            # 清理临时文件
            if os.path.exists(pdb_dir):
                try:
                    shutil.rmtree(pdb_dir)
                    print(f"  -> Cleaned up temporary directory: {pdb_dir}")
                except:
                    pass


def main():
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "decoder_model_path": "./re_llama",  # LLaMA 模型路径
        "model_weights_path": "./model/trained_projectors_stage1_test_name_new_prompt/projectors_and_config_stage1.pt", # 训练好的投影器权重路径
        "protein_mpnn_weights_path": "./ProteinMPNN/vanilla_model_weights/v_48_030.pt", # ProteinMPNN 权重路径
        "xtrimo_model_name": "./xTrimoPGLM", # xTrimoPGLM 模型名或路径

        "struct_embed_dim": 128,   # ProteinMPNN的输出维度
        "seq_embed_dim": 2048,     # xTrimoPGLM的输出维度
        # --- 生成参数 ---
        "max_new_tokens": 150,
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_p": 0.9,
        # "num_beams": 1,
        "do_sample": False,  # 关闭随机采样，使用beam search
        "temperature": 1.0,  # beam search时temperature不起作用
        "top_p": 1.0,       # beam search时top_p不起作用
        "num_beams": 5,     # 使用beam search，平衡质量和确定性
        "early_stopping": True,  # 提前停止，提高效率
        "is_add_name": True,                        
        # "is_add_name": False,                        
        "name_model_name": "allenai/scibert_scivocab_uncased",
        "name_embed_dim": 768,
    }


    # sample_uniprot_id = "Q9CWH0"  
    # sample_sequence = "MSYFGLETFNENQSEENLDEESVILTLVPFKEEEEPNTDYATQSNVSSSTLDHTPPARSLVRHAGIKHPTRTIPSTCPPPSLPPIRDVSRNTLREWCRYHNLSTDGKKVEVYLRLRRHSYSKQECYIPNTSREARMKQGPKKSKIVFRGIGPPSGCQRKKEESGVLEILTSPKESTFAAWARIAMRAAQSMSKNRCPLPSNVEAFLPQATGSRWCVVHGRQLPADKKGWVRLQFLAGQTWVPDTPQRMNFLFLLPACIIPEPGVEDNLLCPECVHSNKKILRNFKIRSRAKKNALPPNMPP"

    sample_uniprot_id = "Q21VW6"  
    sample_sequence = "MNTPQNSPLGQATAYLDQYDASLLFPIARATKRAEIGVTGALPFLGADMWTAFELSWLNLRGKPQVALARFTVPCESPNIIESKSFKLYLNSFNNTRFADVDAVKARLRADLSEAVWRDAGKNVSPDAAAPPSIGVTLLLPELFDREPIYELDGLSLDRLDVECTHYTPAPDLLRVVPDEAPVSEVLVSNLLKSNCPVTGQPDWASVQISYSGAPIDQEGLLQYLVSFRNHNEFHEQCVERIFMDLWTRCKPVRLAVYARYTRRGGLDINPFRTSYAQALPANVRNARQ"
    sample_full_name = "PreQ(0) reductase"

    print("=== Protein Function Prediction Demo ===")
    
    try:
        predictor = ProteinFunctionPredictor(CONFIG)
        print("\nPredictor initialized successfully.\n")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        sys.exit(1)

    try:
        print(f"Predicting function for UniProt ID: {sample_uniprot_id}")
        predicted_function = predictor.predict(sample_uniprot_id, sample_sequence,sample_full_name)
        
        print("\n" + "="*20 + " Predicted Function " + "="*20)
        print(predicted_function)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()