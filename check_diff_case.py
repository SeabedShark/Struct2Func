import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# 读取带分数的CSV文件
df = pd.read_csv('/home/senyutang/protein/results/train_7epochs_best_results_eval_with_scores.csv')

print("检查B7XJI2样本:")
sample = df[df['accession'] == 'B7XJI2'].iloc[0]
print(f"Generated: '{sample['generated']}'")
print(f"Reference: '{sample['reference']}'")
print(f"BLEU: {sample['BLEU']:.6f}")
print(f"BERTScore: {sample['BERTScore_F1']:.6f}")

# 检查是否完全相同
if sample['generated'].strip() == sample['reference'].strip():
    print("\n⚠️ 文本完全相同，但BLEU较低是因为句子太短（只有3个词）")
    print("BLEU默认计算1-4 gram，短句的4-gram匹配会失败，导致分数降低")
    print("BERTScore基于语义相似度，所以能正确识别为1.0")

# 过滤掉完全相同的例子（通过字符串比较）
print("\n" + "="*80)
print("找出真正有差异的多样化例子（排除完全相同的文本）:")
print("="*80)

df['is_identical'] = df['generated'].str.strip() == df['reference'].str.strip()
df_diverse = df[~df['is_identical']].copy()

print(f"排除完全相同的样本后，剩余 {len(df_diverse)} 个有差异的样本")

# 计算综合得分
df_diverse['combined_score'] = (df_diverse['BLEU'] + df_diverse['BERTScore_F1']) / 2

# 1. BERTScore高但BLEU中等的例子（语义相似但表达不同）
print("\n" + "="*80)
print("BERTScore高(>0.85)但BLEU中等(0.3-0.7)的例子 - 语义相似但表达不同:")
print("="*80)
diverse_1 = df_diverse[
    (df_diverse['BERTScore_F1'] > 0.85) & 
    (df_diverse['BLEU'] >= 0.3) & 
    (df_diverse['BLEU'] < 0.7)
].nlargest(10, 'BERTScore_F1')[['accession', 'BLEU', 'BERTScore_F1', 'generated', 'reference']]

for idx, row in diverse_1.iterrows():
    print(f"\nAccession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f} | BERTScore_F1: {row['BERTScore_F1']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 2. BLEU和BERTScore都较高但不等同的例子
print("\n" + "="*80)
print("BLEU和BERTScore都较高(0.7-0.95)但不完全匹配的例子:")
print("="*80)
diverse_2 = df_diverse[
    (df_diverse['BLEU'] >= 0.7) & (df_diverse['BLEU'] < 0.95) &
    (df_diverse['BERTScore_F1'] >= 0.85) & (df_diverse['BERTScore_F1'] < 0.99)
].nlargest(10, 'combined_score')[['accession', 'BLEU', 'BERTScore_F1', 'combined_score', 'generated', 'reference']]

for idx, row in diverse_2.iterrows():
    print(f"\nAccession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f} | BERTScore_F1: {row['BERTScore_F1']:.4f} | 综合得分: {row['combined_score']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 3. 综合得分较高但不是完全匹配的例子
print("\n" + "="*80)
print("综合得分较高(0.8-0.95)的真正多样化例子:")
print("="*80)
diverse_3 = df_diverse[
    (df_diverse['combined_score'] >= 0.8) & (df_diverse['combined_score'] < 0.95)
].nlargest(15, 'combined_score')[['accession', 'BLEU', 'BERTScore_F1', 'combined_score', 'generated', 'reference']]

for idx, row in diverse_3.iterrows():
    print(f"\nAccession: {row['accession']}")
    print(f"BLEU: {row['BLEU']:.4f} | BERTScore_F1: {row['BERTScore_F1']:.4f} | 综合得分: {row['combined_score']:.4f}")
    print(f"Generated: {row['generated']}")
    print(f"Reference: {row['reference']}")

# 保存真正多样化的例子
diverse_cases = pd.concat([
    diverse_1.assign(category='BERTScore高_BLEU中等'),
    diverse_2.assign(category='两者都较高'),
    diverse_3.assign(category='综合得分高')
]).drop_duplicates(subset=['accession'])

output_file = '/home/senyutang/protein/results/train_7epochs_best_results_diverse_cases_filtered.csv'
diverse_cases.to_csv(output_file, index=False)
print(f"\n已保存真正多样化的例子到: {output_file}")
print(f"共找到 {len(diverse_cases)} 个独特的多样化例子")