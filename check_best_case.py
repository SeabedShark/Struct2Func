import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import numpy as np

# 读取CSV文件
df = pd.read_csv('/home/senyutang/protein/results/train_7epochs_best_results_eval.csv')

print(f"总共 {len(df)} 个样本")

# 处理空值：将NaN转换为空字符串，并确保是字符串类型
df['reference'] = df['reference'].fillna('').astype(str)
df['generated'] = df['generated'].fillna('').astype(str)

# 过滤掉完全为空的行
df = df[(df['reference'].str.strip() != '') & (df['generated'].str.strip() != '')]
print(f"过滤空值后剩余 {len(df)} 个有效样本")

print("正在计算BLEU和BERTScore...")

# 计算BLEU分数
smoothing = SmoothingFunction().method1
bleu_scores = []
for idx, row in df.iterrows():
    reference = str(row['reference']).split()
    generated = str(row['generated']).split()
    # 如果reference或generated为空，BLEU为0
    if len(reference) == 0 or len(generated) == 0:
        bleu_scores.append(0.0)
    else:
        bleu = sentence_bleu([reference], generated, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    if (idx + 1) % 500 == 0:
        print(f"已处理 {idx + 1} 个样本的BLEU分数...")

df['BLEU'] = bleu_scores

# 计算BERTScore（批量处理以提高效率）
print("正在计算BERTScore（这可能需要一些时间）...")
P, R, F1 = score(df['generated'].tolist(), 
                 df['reference'].tolist(), 
                 lang='en', 
                 verbose=True,
                 batch_size=32)
df['BERTScore_F1'] = F1.numpy()

# 保存带分数的完整数据
output_file = '/home/senyutang/protein/results/train_7epochs_best_results_eval_with_scores.csv'
df.to_csv(output_file, index=False)
print(f"\n已保存完整结果到: {output_file}")

# 找出BLEU最高的前10个样本
print("\n" + "="*80)
print("BLEU分数最高的前10个样本:")
print("="*80)
top_bleu = df.nlargest(10, 'BLEU')[['accession', 'BLEU', 'BERTScore_F1', 'generated', 'reference']]
for idx, row in top_bleu.iterrows():
    rank = len(top_bleu) - list(top_bleu.index).index(idx)
    print(f"\n排名 {rank}")
    print(f"Accession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f}")
    print(f"BERTScore_F1: {row['BERTScore_F1']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 找出BERTScore最高的前10个样本
print("\n" + "="*80)
print("BERTScore_F1分数最高的前10个样本:")
print("="*80)
top_bertscore = df.nlargest(10, 'BERTScore_F1')[['accession', 'BLEU', 'BERTScore_F1', 'generated', 'reference']]
for idx, row in top_bertscore.iterrows():
    rank = len(top_bertscore) - list(top_bertscore.index).index(idx)
    print(f"\n排名 {rank}")
    print(f"Accession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f}")
    print(f"BERTScore_F1: {row['BERTScore_F1']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 找出综合得分最高的样本（BLEU和BERTScore的平均值）
df['combined_score'] = (df['BLEU'] + df['BERTScore_F1']) / 2
print("\n" + "="*80)
print("综合得分（BLEU和BERTScore平均值）最高的前10个样本:")
print("="*80)
top_combined = df.nlargest(10, 'combined_score')[['accession', 'BLEU', 'BERTScore_F1', 'combined_score', 'generated', 'reference']]
for idx, row in top_combined.iterrows():
    rank = len(top_combined) - list(top_combined.index).index(idx)
    print(f"\n排名 {rank}")
    print(f"Accession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f}")
    print(f"BERTScore_F1: {row['BERTScore_F1']:.4f}")
    print(f"综合得分: {row['combined_score']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 保存top结果
top_results_file = '/home/senyutang/protein/results/train_7epochs_best_results_top_cases.csv'
top_combined.to_csv(top_results_file, index=False)
print(f"\n已保存top结果到: {top_results_file}")