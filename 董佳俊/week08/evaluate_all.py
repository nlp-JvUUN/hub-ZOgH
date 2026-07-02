"""
作业：加载已训练的 6 个 checkpoint，在验证集上评估

完全复用 src/evaluate.py 的 eval_biencoder / eval_crossencoder 模块。
额外产出：BiEncoder 的相似度分布图（正 vs 负样本分离）

使用方式: python evaluate_all.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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

from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

# ── 6 组实验配置（checkpoint 已训练完毕）────────────────────────────────────
EXPERIMENTS = [
    {"dataset": "lcqmc", "method": "biencoder_cosine",   "type": "biencoder"},
    {"dataset": "lcqmc", "method": "biencoder_triplet",  "type": "biencoder"},
    {"dataset": "lcqmc", "method": "crossencoder",       "type": "crossencoder"},
    {"dataset": "bq_corpus", "method": "biencoder_cosine",  "type": "biencoder"},
    {"dataset": "bq_corpus", "method": "biencoder_triplet", "type": "biencoder"},
    {"dataset": "bq_corpus", "method": "crossencoder",      "type": "crossencoder"},
]


def plot_similarity_distribution(sims, labels, threshold, save_path, title):
    """正/负样本相似度分布直方图（BiEncoder 专属）"""
    sims   = np.array(sims)
    labels = np.array(labels)
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_sims, bins=50, alpha=0.6, label=f"正样本 (n={len(pos_sims)})",
            color="#2196F3", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label=f"负样本 (n={len(neg_sims)})",
            color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"最优阈值 = {threshold:.2f}")
    ax.set_xlabel("余弦相似度")
    ax.set_ylabel("密度")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  相似度分布图 -> {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    all_eval_logs = []

    for exp in EXPERIMENTS:
        ds_name = exp["dataset"]
        method  = exp["method"]
        mtype   = exp["type"]
        ckpt_path = RESULTS_DIR / ds_name / method / "best_model.pt"

        if not ckpt_path.exists():
            print(f"\n[SKIP] checkpoint 不存在: {ckpt_path}")
            continue

        print(f"\n{'='*55}")
        print(f"  评估: {ds_name} / {method}")
        print(f"{'='*55}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})

        data_path = DATA_DIR / ds_name / "validation.jsonl"

        if mtype == "biencoder":
            model = build_biencoder(
                bert_path=str(BERT_PATH),
                pool=saved_args.get("pool", "mean"),
                num_hidden_layers=saved_args.get("num_hidden_layers"),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            ds     = PairDataset(data_path, tokenizer, saved_args.get("max_length_bi", 64))
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            metrics = eval_biencoder(model, loader, device)

            # 生成相似度分布图
            fig_path = FIG_DIR / ds_name / f"{method}_sim_dist.png"
            plot_similarity_distribution(
                metrics["similarities"], metrics["labels"], metrics["threshold"],
                fig_path,
                title=f"{ds_name.upper()} / {method} 相似度分布（validation）",
            )

            print(f"  Threshold={metrics['threshold']:.2f}  "
                  f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

            all_eval_logs.append({
                "dataset": ds_name, "method": method,
                "val_acc": metrics["accuracy"], "val_f1": metrics["f1"],
                "threshold": metrics["threshold"], "auc": metrics["auc"],
            })

        else:  # crossencoder
            model = build_crossencoder(
                bert_path=str(BERT_PATH),
                num_hidden_layers=saved_args.get("num_hidden_layers"),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            metrics = eval_crossencoder(model, loader, device)

            print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

            all_eval_logs.append({
                "dataset": ds_name, "method": method,
                "val_acc": metrics["accuracy"], "val_f1": metrics["f1"],
            })

    # 保存评估汇总
    eval_summary_path = HOMEWORK_DIR / "eval_summary.json"
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_eval_logs, f, ensure_ascii=False, indent=2)
    print(f"\n评估汇总已保存 -> {eval_summary_path}")


if __name__ == "__main__":
    main()
