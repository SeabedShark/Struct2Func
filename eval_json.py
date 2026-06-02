#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
import evaluate
from typing import List

def truncate_texts(texts, max_len=512):
    return [t[:max_len] for t in texts]

def load_predictions(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = [item["response"].strip() for item in data]
    refs = [item["target"].strip() for item in data]
    return preds, refs

def avg(x):
    return sum(x) / len(x) if isinstance(x, (list, tuple)) and len(x) > 0 else float('nan')

def main():
    json_path = "./Mol-Instructions/output/protein_function_result.json"  # ← 请根据实际情况修改路径
    save_metrics_csv = "./prollama_metrics.csv"

    preds, refs = load_predictions(json_path)
    preds = truncate_texts(preds)
    refs = truncate_texts(refs)

    if len(preds) != len(refs):
        raise ValueError("预测与参考数量不一致！")

    print(f"共加载 {len(preds)} 条样本，开始评估...")

    # 加载指标（与 eval.txt 一致）
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    # 计算指标
    res_bleu = bleu.compute(predictions=preds, references=refs)
    res_rouge = rouge.compute(predictions=preds, references=refs)
    res_bert = bert.compute(
        predictions=preds,
        references=refs,
        model_type="dmis-lab/biobert-large-cased-v1.1",
        num_layers=24,
    )

    # 打印结果
    print("==== Metrics ====")
    bleu_score = res_bleu.get('bleu', res_bleu) if isinstance(res_bleu, dict) else res_bleu
    print(f"BLEU: {bleu_score}")
    print(f"ROUGE-1: {res_rouge.get('rouge1'):.4f}")
    print(f"ROUGE-2: {res_rouge.get('rouge2'):.4f}")
    print(f"ROUGE-L: {res_rouge.get('rougeL'):.4f}")
    print(f"BERTScore F1 (BioBERT): {avg(res_bert['f1']):.4f}")

    # 保存指标到 CSV
    row = {
        "num_samples": len(preds),
        "BLEU": float(bleu_score) if isinstance(bleu_score, (int, float)) else float('nan'),
        "ROUGE-1": float(res_rouge.get('rouge1', float('nan'))),
        "ROUGE-2": float(res_rouge.get('rouge2', float('nan'))),
        "ROUGE-L": float(res_rouge.get('rougeL', float('nan'))),
        "BERTScore_F1": float(avg(res_bert.get('f1', [])))
    }

    mdf = pd.DataFrame([row])
    mdf.to_csv(save_metrics_csv, index=False)
    print(f"[Metrics saved] {save_metrics_csv}")

if __name__ == "__main__":
    main()