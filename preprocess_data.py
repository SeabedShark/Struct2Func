import os
import sys
import requests
import warnings
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# --- 配置部分 ---
# 请根据您的实际路径进行修改
CONFIG = {
    "data_save_path": "./data",
    "split": "train",

    "csv_path": "./data/train.csv",  # 您的CSV文件路径 
    "protein_mpnn_repo_path": "./ProteinMPNN",  # ProteinMPNN仓库的路径
    "protein_mpnn_weights_path": "./ProteinMPNN/vanilla_model_weights/v_48_030.pt", # ProteinMPNN权重路径
    "xtrimo_model_name": "./xTrimoPGLM",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    version: int = 4
    ):
    
    os.makedirs(out_dir, exist_ok=True)
    pdb_filename = f"AF-{uniprot_id.upper()}-F1-model_v{version}.pdb"
    pdb_filepath = os.path.join(out_dir, pdb_filename)

    if os.path.exists(pdb_filepath):
        # print(f"Structure for {uniprot_id} already exists. Skipping download.")
        return pdb_filepath

    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    query_url = f"{BASE_URL}{pdb_filename}"

    try:
        response = requests.get(query_url, stream=True)
        response.raise_for_status()  # 如果请求失败 (如 404), 会抛出HTTPError

        with open(pdb_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # print(f"Successfully downloaded {pdb_filename}")
        return pdb_filepath
    except requests.exceptions.RequestException as e:
        warnings.warn(f"Could not download structure for {uniprot_id}: {e}")
        return None


# --- ProteinMPNN 嵌入提取模块 ---
# 将ProteinMPNN仓库路径添加到sys.path以便导入
sys.path.append(CONFIG["protein_mpnn_repo_path"])
from ProteinMPNN.protein_mpnn_utils import  parse_PDB, StructureDatasetPDB, tied_featurize, ProteinMPNN, gather_nodes 

class ProteinMPNNEmbedder(torch.nn.Module):
    """
    一个包装器,用于加载ProteinMPNN并提取其结构嵌入。
    """
    def __init__(self, weights_path, device):
        super().__init__()
        self.device = device
        
        # 加载模型定义和权重
        checkpoint = torch.load(weights_path, map_location=device)
        self.model = ProteinMPNN(
            num_letters=21, 
            node_features=128, 
            edge_features=128,
            hidden_dim=128, 
            num_encoder_layers=3, 
            num_decoder_layers=3,
            k_neighbors=checkpoint['num_edges']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("ProteinMPNN model loaded successfully.")

    @torch.no_grad()
    def stru_embed(self, pdb_path):
        """
        从PDB文件路径生成结构嵌入。
        """
        # 1. 使用ProteinMPNN的工具函数解析和特征化PDB
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found at {pdb_path}")
        
        protein_dict_list = parse_PDB(pdb_path)
        if not protein_dict_list:
          raise ValueError(f"Could not parse PDB file: {pdb_path}")   
      
        batch = tied_featurize([protein_dict_list[0]], self.device, None)
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = batch
        
        with torch.no_grad():
            # 调用我们动态添加的方法来获取 h_V
            h_V = self.model.get_encoder_embeddings(X, mask, residue_idx, chain_encoding_all)

        h_V_numpy = h_V[0].cpu().numpy()
        mask_numpy = mask[0].cpu().numpy()
        
        return h_V_numpy, mask_numpy

        


# --- xTrimoPGLM 嵌入提取模块 ---
from transformers import AutoTokenizer, AutoModelForMaskedLM

class XtrimoPGLMEmbedder(torch.nn.Module):
    """
    一个包装器,用于加载xTrimoPGLM并提取其序列嵌入。
    """
    def __init__(self, model_name="biomap-research/proteinglm-1b-mlm", device="cuda:0"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 # 使用bfloat16以节省显存
        ).to(device)
        self.model.eval()
        print("xTrimoPGLM model loaded successfully.")

    @torch.no_grad()
    def seq_embed(self, sequence):
        """
        从氨基酸序列字符串生成序列嵌入。
        """
        inputs = self.tokenizer(sequence, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs,output_hidden_states=True,return_last_hidden_state=True)
        last_hidden_states = outputs.hidden_states
        last_hidden_states = last_hidden_states.to(torch.float32)
        # 去除eos token
        embeddings = last_hidden_states.squeeze(1)[:-1, :]
        # print(embeddings.shape)
        mask = torch.ones(embeddings.shape[0], dtype=torch.float32)
        
        return embeddings.cpu().numpy(), mask.cpu().numpy()


# --- 主预处理流程 ---
def main():
    # 创建输出目录
    pdb_dir = os.path.join(CONFIG["data_save_path"], CONFIG["split"], "pdb","pdb")#pdb/pdb
    struct_embed_dir = os.path.join(CONFIG["data_save_path"], CONFIG["split"], "processed", "struct")
    seq_embed_dir = os.path.join(CONFIG["data_save_path"], CONFIG["split"], "processed", "seq")

    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(struct_embed_dir, exist_ok=True)
    os.makedirs(seq_embed_dir, exist_ok=True)

    print('downloading the data:\n')
    # 加载CSV
    df = pd.read_csv(CONFIG["csv_path"])
    for prot in tqdm(set(df.AlphaFoldDB)):
        if os.path.exists(os.path.join(pdb_dir, 'AF-'+str(prot)+'-F1-model_v4.pdb')):
            continue
        download_alphafold_structure(uniprot_id=str(prot), out_dir=pdb_dir)

    # 实例化嵌入器
    device = CONFIG["device"]
    mpnn_embedder = ProteinMPNNEmbedder(CONFIG["protein_mpnn_weights_path"], device)
    pglm_embedder = XtrimoPGLMEmbedder(CONFIG["xtrimo_model_name"],device=device)

    # 遍历数据
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        accession_id = row['accession']
        sequence = row['sequence']
        
        pdb = os.path.join(pdb_dir,f"AF-{accession_id}-F1-model_v4.pdb")
        struct_embed_path = os.path.join(struct_embed_dir, f"{accession_id}.pt")
        seq_embed_path = os.path.join(seq_embed_dir, f"{accession_id}.pt")

        if not os.path.exists(pdb):
            warnings.warn(f"未找到PDB文件 {accession_id}, 跳过。")
            continue

        # 如果两种嵌入都已存在，则跳过
        if os.path.exists(struct_embed_path) and os.path.exists(seq_embed_path):
            continue

        try:
            # 2. 计算并保存ProteinMPNN结构嵌入
            if not os.path.exists(struct_embed_path):
                struct_embedding,struct_mask = mpnn_embedder.stru_embed(pdb)
                torch.save({
                    'embedding': torch.from_numpy(struct_embedding),
                    'mask': torch.from_numpy(struct_mask)
                }, struct_embed_path)
                print(f"Saved ProteinMPNN embedding for {accession_id}")

            # 3. 计算并保存xTrimoPGLM序列嵌入
            if not os.path.exists(seq_embed_path):
                seq_embedding,seq_mask= pglm_embedder.seq_embed(sequence)
                torch.save({
                    'embedding': torch.from_numpy(seq_embedding),
                    'mask': torch.from_numpy(seq_mask)
                }, seq_embed_path)
                print(f"Saved xTrimoPGLM embedding for {accession_id}")

        except Exception as e:
            print(f"An error occurred while processing {accession_id}: {e}")
            continue

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()