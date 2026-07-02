"""
作业：Bad Case 分析（最佳方法：CrossEncoder）

完全复用 src/analyze_badcases.py 的分析逻辑。
对每个数据集的最佳方法做 FP/FN 分析 + 语言特征 + Score 分布图。

使用方式: python analyze_badcases.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# ── 路径 ────────────────────────────────────────────────────────────────────
HOMEWORK_DIR = Path(__file__).parent
PROJECT_ROOT = HOMEWORK_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
BERT_PATH    = Path("D:/八斗学习内容/pretrain_models/bert-base-chinese")
RESULTS_DIR  = HOMEWORK_DIR / "results"
FIG_DIR      = HOMEWORK_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))

from dataset import CrossEncoderDataset, load_jsonl
from model import build_crossencoder


@torch.no_grad()
def collect_crossencoder_preds(model, loader, raw_rows, device):
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        ).cpu()
        probs  = torch.softmax(logits, dim=-1)[:, 1].tolist()
        preds  = logits.argmax(dim=-1).tolist()
        labels = batch["label"].tolist()

        for prob, pred, label in zip(probs, preds, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": prob,
                "pred":  pred,
            })
            idx += 1
    return results


def split_badcases(results, threshold=0.5):
    fp_high, fp_border = [], []
    fn_high, fn_border = [], []

    for r in results:
        if r["pred"] == r["label"]:
            continue
        gap = abs(r["score"] - threshold)
        if r["pred"] == 1 and r["label"] == 0:   # FP
            (fp_high if gap > 0.15 else fp_border).append(r)
        else:                                      # FN
            (fn_high if gap > 0.15 else fn_border).append(r)

    fp_high.sort(key=lambda x: x["score"], reverse=True)
    fn_high.sort(key=lambda x: x["score"])

    return {"fp_high": fp_high, "fp_border": fp_border,
            "fn_high": fn_high, "fn_border": fn_border}


def analyze_patterns(cases, label):
    if not cases:
        return
    len_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"])) for r in cases]
    lens_s1   = [len(r["sentence1"]) for r in cases]
    lens_s2   = [len(r["sentence2"]) for r in cases]

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)

    jaccards = [jaccard(r["sentence1"], r["sentence2"]) for r in cases]

    print(f"\n  【{label}】共 {len(cases)} 条")
    print(f"    长度差     : 均值={np.mean(len_diffs):.1f}  中位={np.median(len_diffs):.0f}")
    print(f"    字符 Jaccard: 均值={np.mean(jaccards):.3f}")


def print_cases(cases, title, n=5):
    print(f"\n  {title} (展示 {min(n, len(cases))} 条)：")
    for r in cases[:n]:
        print(f"    score={r['score']:.3f}  | {r['sentence1']}")
        print(f"    {'':>12}  | {r['sentence2']}")
        print()


def plot_score_dist_with_errors(results, save_path, ds_label):
    scores  = np.array([r["score"]  for r in results])
    labels  = np.array([r["label"]  for r in results])
    correct = np.array([r["pred"] == r["label"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左：正/负样本分数分布
    ax = axes[0]
    ax.hist(scores[labels==1], bins=40, alpha=0.6, label="positive",
            color="#2196F3", density=True)
    ax.hist(scores[labels==0], bins=40, alpha=0.6, label="negative",
            color="#F44336", density=True)
    ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.50")
    ax.set_xlabel("Score (P(similar))")
    ax.set_title(f"[{ds_label}] Positive vs Negative Distribution")
    ax.legend(fontsize=8)

    # 右：正确 vs 错误
    ax = axes[1]
    ax.hist(scores[correct],  bins=40, alpha=0.6, label="correct",
            color="#4CAF50", density=True)
    ax.hist(scores[~correct], bins=40, alpha=0.6, label="error",
            color="#F44336", density=True)
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Score")
    ax.set_title(f"[{ds_label}] Correct vs Error Score Distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 -> {save_path}")


def print_optimization_directions(fp_jaccard_mean, fn_jaccard_mean, ds_label):
    print(f"\n{'='*60}")
    print(f"优化方向建议 [{ds_label}]")
    print(f"{'='*60}")

    if fp_jaccard_mean > 0.5:
        print("""
[1] 针对 FP（词汇重叠高但语义不同）
  - 难负样本挖掘：用训练好的模型在语料中挖掘相似度高但标签=0 的对
  - 增大 CosineEmbeddingLoss 的 margin（0.3 -> 0.5）""")
    else:
        print("""
[1] 针对 FP（语义理解不足）
  - 增加 BERT 层数（4 -> 12）
  - 引入领域预训练模型""")

    if fn_jaccard_mean < 0.3:
        print("""
[2] 针对 FN（同义异词）
  - SimCSE 对比学习预训练
  - 正样本数据增强（同义改写）""")
    else:
        print("""
[2] 针对 FN（临界错误）
  - 阈值微调可改善部分错误
  - 增加训练 epoch""")

    print("""
[3] 工程层面
  - 两阶段级联：BiEncoder 召回 Top-K -> CrossEncoder 精排
  - 阈值按实际流量分布校准""")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 对每个数据集的最佳方法（CrossEncoder）做 Bad Case 分析
    for ds_name in ["lcqmc", "bq_corpus"]:
        print(f"\n{'#'*60}")
        print(f"  Bad Case 分析: {ds_name} / CrossEncoder")
        print(f"{'#'*60}")

        ckpt_path = RESULTS_DIR / ds_name / "crossencoder" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  checkpoint 不存在: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})

        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / ds_name / "validation.jsonl"
        raw_rows  = load_jsonl(data_path)
        print(f"  数据集: {data_path.name}  共 {len(raw_rows):,} 条")

        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        results = collect_crossencoder_preds(model, loader, raw_rows, device)

        # 整体指标
        labels    = [r["label"] for r in results]
        preds     = [r["pred"]  for r in results]
        n_correct = sum(1 for l, p in zip(labels, preds) if l == p)
        accuracy  = n_correct / len(results)
        print(f"  整体准确率: {accuracy:.4f}  错误数: {len(results)-n_correct}")

        # 分类 bad case
        badcases = split_badcases(results, 0.5)
        fp_all   = badcases["fp_high"] + badcases["fp_border"]
        fn_all   = badcases["fn_high"] + badcases["fn_border"]

        print(f"\n  Bad Case 汇总:")
        print(f"    FP 假阳性: {len(fp_all):>4} 条  "
              f"(高置信:{len(badcases['fp_high'])}, 临界:{len(badcases['fp_border'])})")
        print(f"    FN 假阴性: {len(fn_all):>4} 条  "
              f"(高置信:{len(badcases['fn_high'])}, 临界:{len(badcases['fn_border'])})")

        # 语言特征分析
        print(f"\n  语言特征分析：")
        analyze_patterns(fp_all, "FP（假阳性）")
        analyze_patterns(fn_all, "FN（假阴性）")

        fp_jaccard = np.mean([
            len(set(r["sentence1"]) & set(r["sentence2"])) /
            max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
            for r in fp_all
        ]) if fp_all else 0
        fn_jaccard = np.mean([
            len(set(r["sentence1"]) & set(r["sentence2"])) /
            max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
            for r in fn_all
        ]) if fn_all else 0

        # 典型案例
        print_cases(badcases["fp_high"],  "FP 高置信度错误（score 最高）", n=5)
        print_cases(badcases["fn_high"],  "FN 高置信度错误（score 最低）", n=5)

        # 可视化
        plot_score_dist_with_errors(results,
                                    FIG_DIR / ds_name / "crossencoder_badcase_dist.png",
                                    ds_name.upper())

        # 优化方向
        print_optimization_directions(fp_jaccard, fn_jaccard, ds_name.upper())


if __name__ == "__main__":
    main()
