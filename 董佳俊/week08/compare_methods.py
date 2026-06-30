"""
作业：每个数据集上三种方法的对比

复用 src/compare_methods.py 的逻辑：
  - 柱状对比图（Acc + F1）
  - BiEncoder 的相似度分布叠放对比

使用方式: python compare_methods.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

from dataset import PairDataset
from evaluate import eval_biencoder
from model import build_biencoder

METHODS = [
    {
        "key": "biencoder_cosine",  "label": "BiEncoder\n(CosineLoss)",
        "type": "biencoder", "color": "#2196F3",
    },
    {
        "key": "biencoder_triplet", "label": "BiEncoder\n(TripletLoss)",
        "type": "biencoder", "color": "#4CAF50",
    },
    {
        "key": "crossencoder", "label": "CrossEncoder\n(CrossEntropyLoss)",
        "type": "crossencoder", "color": "#FF9800",
    },
]


def get_eval_result(ds_name, method, mtype, tokenizer, device):
    """加载 checkpoint 并评估，返回 metrics dict"""
    ckpt_path = RESULTS_DIR / ds_name / method / "best_model.pt"

    # 优先读 log.json（训练时已保存 eval 结果）
    log_path = RESULTS_DIR / ds_name / method / "log.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        return {
            "accuracy": log["val_acc"],
            "f1":       log["val_f1"],
            "threshold": log.get("threshold", None),
            "similarities": None,
            "labels": None,
        }

    # fallback: 重新评估（log 不存在时）
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
        return {
            "accuracy": metrics["accuracy"],
            "f1":       metrics["f1"],
            "threshold": metrics["threshold"],
            "similarities": metrics["similarities"],
            "labels": metrics["labels"],
        }
    else:
        # CrossEncoder: log.json 已包含所有需要的数据
        return {"accuracy": log["val_acc"], "f1": log["val_f1"], "threshold": None,
                "similarities": None, "labels": None}


def plot_comparison_bar(results, save_path, dataset_name):
    """柱状图：Accuracy + F1 对比"""
    names  = [r["label"] for r in results]
    accs   = [r["accuracy"] for r in results]
    f1s    = [r["f1"] for r in results]
    colors = [r["color"] for r in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors,
                   alpha=0.5, hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Method Comparison on {dataset_name.upper()} Validation (4-layer BERT, 1 epoch)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  柱状对比图 -> {save_path}")


def plot_sim_distributions(biencoder_results, save_path, dataset_name):
    """BiEncoder 方法间的相似度分布对比"""
    bi_list = [r for r in biencoder_results
               if r.get("similarities") is not None]
    n = len(bi_list)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, bi_list):
        sims   = np.array(r["similarities"])
        labels = np.array(r["labels"])
        ax.hist(sims[labels==1], bins=40, alpha=0.6, label="positive",
                color="#2196F3", density=True)
        ax.hist(sims[labels==0], bins=40, alpha=0.6, label="negative",
                color="#F44336", density=True)
        ax.axvline(r["threshold"], color="black", linestyle="--",
                   label=f"threshold={r['threshold']:.2f}")
        ax.set_title(r["label"].replace("\n", " "))
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=8)

    fig.suptitle(f"BiEncoder Similarity Distribution ({dataset_name.upper()})", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  相似度分布对比 -> {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    for ds_name in ["lcqmc", "bq_corpus"]:
        print(f"\n{'='*55}")
        print(f"  数据集: {ds_name}")
        print(f"{'='*55}")

        results = []
        for m in METHODS:
            print(f"  加载 {m['key']} ...")
            metrics = get_eval_result(ds_name, m["key"], m["type"], tokenizer, device)
            metrics.update({
                "label": m["label"], "color": m["color"],
                "key": m["key"], "type": m["type"],
            })
            results.append(metrics)

        # 控制台对比表
        print(f"\n  {'方法':<30} {'Acc':>8} {'F1':>8} {'阈值':>8}")
        print(f"  {'-'*54}")
        for r in results:
            thr_str = f"{r['threshold']:.2f}" if r.get("threshold") else "argmax"
            print(f"  {r['key']:<28} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {thr_str:>8}")

        best = max(results, key=lambda x: x["f1"])
        print(f"\n  最佳方法: {best['key']}  F1={best['f1']:.4f}")

        # 可视化
        plot_comparison_bar(results, FIG_DIR / ds_name / "method_comparison_bar.png",
                            ds_name)

        bi_results = [r for r in results if r["type"] == "biencoder"
                      and r.get("similarities") is not None]
        if bi_results:
            # 如果没有 similarities，快速推理获取
            if all(r.get("similarities") is None for r in results if r["type"] == "biencoder"):
                print("  (BiEncoder 无相似度数据，补充推理...)")
                # reload with similarities
                for i, r in enumerate(results):
                    if r["type"] == "biencoder":
                        fresh = get_eval_result_with_sims(
                            ds_name, r["key"], tokenizer, device)
                        r["similarities"] = fresh["similarities"]
                        r["labels"] = fresh["labels"]
                        results[i] = r
                bi_results = [r for r in results if r["type"] == "biencoder"
                              and r.get("similarities") is not None]
            if bi_results:
                plot_sim_distributions(bi_results,
                                       FIG_DIR / ds_name / "biencoder_sim_distributions.png",
                                       ds_name)


def get_eval_result_with_sims(ds_name, method, tokenizer, device):
    """强制重新推理获取相似度分布（仅 BiEncoder）"""
    ckpt_path = RESULTS_DIR / ds_name / method / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    model = build_biencoder(
        bert_path=str(BERT_PATH),
        pool=saved_args.get("pool", "mean"),
        num_hidden_layers=saved_args.get("num_hidden_layers"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_path = DATA_DIR / ds_name / "validation.jsonl"
    ds     = PairDataset(data_path, tokenizer, saved_args.get("max_length_bi", 64))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    metrics = eval_biencoder(model, loader, device)
    return {
        "accuracy": metrics["accuracy"], "f1": metrics["f1"],
        "threshold": metrics["threshold"],
        "similarities": metrics["similarities"], "labels": metrics["labels"],
    }


if __name__ == "__main__":
    main()
