import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer,LlamaTokenizer
import sys
import random
# --- 1. 自定义 ProteinDataset 类 ---

class ProteinDataset(Dataset):

    def __init__(self, csv_path, processed_dir, split):
        """
        初始化数据集。
        
        参数:
        - csv_path: CSV文件路径。
        - processed_dir (str): 存放预处理数据的根目录 (例如, './data')。
        - split (str): 数据集划分的名称 (例如, 'test_12samples')。
        """
        self.df = pd.read_csv(csv_path)
        self.struct_embed_dir = os.path.join(processed_dir, split, "processed", "struct")
        self.seq_embed_dir = os.path.join(processed_dir, split, "processed", "seq")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据索引获取一个数据样本。
        """
        row = self.df.iloc[idx]
        accession_id = row['accession']
        function_description = row['function']

        struct_processed_path = os.path.join(self.struct_embed_dir, f"{accession_id}.pt")
        seq_processed_path = os.path.join(self.seq_embed_dir, f"{accession_id}.pt")

        # 检查文件是否存在，如果不存在则跳过该样本
        if not os.path.exists(struct_processed_path) or not os.path.exists(seq_processed_path):
            # 返回None，collate_fn会处理这种情况
            return None

        # 加载包含 embedding 和 mask 的字典
        struct_data = torch.load(struct_processed_path)
        seq_data = torch.load(seq_processed_path)
        full_name = row['Full Name']

        return {
            "accession": accession_id,
            "struct_embedding": struct_data['embedding'],
            "struct_mask": struct_data['mask'],
            "seq_embedding": seq_data['embedding'],
            "seq_mask": seq_data['mask'],
            "function_description": function_description,
            "full_name": full_name
        }


def custom_collate_fn(batch, tokenizer, name_encoder=None, use_name=False, name_dropout_p=0.0):
    """
    自定义的collate函数,用于处理变长数据并进行填充。
    """
    # 过滤掉在 __getitem__ 中返回None的无效样本
    batch = [item for item in batch if item is not None]
    # 长度不一致的样本直接跳过（最小改动，避免后续相加报错）
    filtered = []
    for item in batch:
        try:
            if item['struct_embedding'].size(0) == item['seq_embedding'].size(0):
                filtered.append(item)
        except Exception:
            # 任意异常视为不合法样本
            continue
    batch = filtered
    if not batch:
        return None

    # <protein> 特殊的占位符
    instruction = "Analyze the following amino acid sequence, and determine the function of the resulting protein, its subcellular localization, and any biological processes it may be part of:"
    instruction_text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n<protein>\n\n### Response:\n"

    
    instruction_texts = [instruction_text] * len(batch)
    answer_texts = [item['function_description'] for item in batch]
    accession_list = [item.get('accession', '') for item in batch]

    # 我们需要确保在答案文本的末尾添加一个结束符（eos_token），
    instruction_tokens = tokenizer(instruction_texts, return_tensors="pt", padding=False, truncation=False)
    answer_tokens = tokenizer(
        [text + tokenizer.eos_token for text in answer_texts], 
        return_tensors="pt", 
        padding=True, # 对答案进行填充
        truncation=True,
        max_length=512 # 设置一个最大长度
    )

    # 对嵌入和掩码进行填充
    struct_embeddings = [item['struct_embedding'] for item in batch]
    struct_masks = [item['struct_mask'] for item in batch]
    seq_embeddings = [item['seq_embedding'] for item in batch]
    seq_masks = [item['seq_mask'] for item in batch]

    # 使用pad_sequence进行填充，batch_first=True表示批次维度在前
    padded_struct_embeds = pad_sequence(struct_embeddings, batch_first=True, padding_value=0.0)
    padded_struct_masks = pad_sequence(struct_masks, batch_first=True, padding_value=0.0)
    padded_seq_embeds = pad_sequence(seq_embeddings, batch_first=True, padding_value=0.0)
    padded_seq_masks = pad_sequence(seq_masks, batch_first=True, padding_value=0.0)

    out = {
        "struct_embeds": padded_struct_embeds,
        "struct_masks": padded_struct_masks,
        "seq_embeds": padded_seq_embeds,
        "seq_masks": padded_seq_masks,
        "instruction_tokens": instruction_tokens,
        "answer_tokens": answer_tokens,
        "accession_list": accession_list
    }

    if use_name and (name_encoder is not None):
        def to_safe_str(x):
            if isinstance(x, str):
                return x.strip()
            if x is None:
                return ""
            if isinstance(x, float):
                try:
                    if math.isnan(x):
                        return ""
                except Exception:
                    pass
            return str(x).strip()

        names = [to_safe_str(b.get("full_name", "")) for b in batch]
        if name_dropout_p > 0:
            names = [("" if (random.random() < name_dropout_p) else n) for n in names]
        with torch.no_grad():
            name_pooled = name_encoder.encode(names)  # [B,1,D_name]
        out["name_embeds"] = name_pooled
    
    out["struct_embeds"] = out["struct_embeds"].to(dtype=torch.bfloat16)
    out["seq_embeds"] = out["seq_embeds"].to(dtype=torch.bfloat16)
    if "name_embeds" in out:
        out["name_embeds"] = out["name_embeds"].to(dtype=torch.bfloat16)

    return out



if __name__ == '__main__':

    CONFIG = {
        "data_save_path": "./data",
        "split": "test24",
        "csv_path": "./data/test24.csv",
        "decoder_model_name": "./re_llama" 
    }

    # 加载LLaMA的分词器

    try:
        tokenizer = LlamaTokenizer.from_pretrained(CONFIG["decoder_model_name"], use_fast=True)
        tokenizer.padding_side = 'left'

        if tokenizer.pad_token is None:
            print("Warning: tokenizer.pad_token is None in dataset.py test.")
        # 方法1: 使用 eos_token 作为 pad_token (常见做法，尤其对于 LLaMA)
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
            else:
                # 方法2: 添加一个新的特殊 token 作为 pad_token
                # 注意：如果模型没有为这个新token预留嵌入维度，可能需要 resize_token_embeddings，
                # 但在 dataset.py 的测试中通常不需要，因为我们不训练模型。
                # 但在 train.py 中如果添加了新token，则需要调整模型。
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token = '[PAD]'
                print("Added new '[PAD]' token and set it as tokenizer.pad_token.")
                # 注意：在 dataset.py 测试中，通常不涉及模型，所以不调用 resize_token_embeddings
        else:
            print(f"tokenizer.pad_token already set to: {tokenizer.pad_token}")
    except Exception as e:
        print(f"无法加载分词器: {e}")
        print("请确保你登录Hugging Face CLI并且可以访问该模型。")
        sys.exit(1)
        
    # 为collate_fn添加tokenizer
    from functools import partial
    collate_fn_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer)

    # 实例化数据集
    dataset = ProteinDataset(
        csv_path=CONFIG["csv_path"],
        processed_dir=CONFIG["data_save_path"],
        split=CONFIG["split"]
    )

    # 实例化数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4, # 批次大小可以根据你的显存调整
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer
    )

    print(f"数据集大小: {len(dataset)}")
    print("开始迭代DataLoader以进行测试...")
   
        # 迭代一个批次进行测试
    try:
        batch_data = next(iter(dataloader))
        
        print("\n成功获取一个批次的数据")
        print("批次中包含的键:", batch_data.keys())
        
        print("\n数据维度检查:")
        print(f"结构嵌入 (struct_embeds) shape: {batch_data['struct_embeds'].shape}")
        print(f"结构掩码 (struct_masks) shape: {batch_data['struct_masks'].shape}")
        print(f"序列嵌入 (seq_embeds) shape: {batch_data['seq_embeds'].shape}")
        print(f"序列掩码 (seq_masks) shape: {batch_data['seq_masks'].shape}")
        print(f"指令 (instruction_tokens) input_ids shape: {batch_data['instruction_tokens']['input_ids'].shape}")
        print(f"答案 (answer_tokens) input_ids shape: {batch_data['answer_tokens']['input_ids'].shape}")
        
        print("\n数据管道构建成功")

    except StopIteration:
        print("DataLoader为空,请检查数据集是否正确加载或文件路径是否正确。")
    except Exception as e:
        print(f"\n在迭代DataLoader时发生错误: {e}")
        import traceback
        traceback.print_exc()