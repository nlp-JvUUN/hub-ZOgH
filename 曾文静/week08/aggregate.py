# -*- coding: utf-8 -*-
"""
aggregate.py — 汇总三个数据集 × 三种方法的训练日志，生成对比表和对比图

用法：
  python aggregate.py

读取 results/{afqmc,lcqmc,bq_corpus}/logs/*.json（afqmc 为老师已跑好的基线日志）
输出：
  results/comparison_table.md   对比表（markdown）
  results/figures/f1_comparison.png      3数据集×3方法 柱状图
  results/figures/f1_epochs.png          各数据集 F1 随 epoch 曲线（3 子图）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.resolve()
RESULTS = ROOT / "results"
DATASETS = ["afqmc", "lcqmc", "bq_corpus"]
METHODS = [
    ("biencoder_cosine_log.json", "BiEncoder+Cosine"),
    ("biencoder_triplet_log.json", "BiEncoder+Triplet"),
    ("crossencoder_log.json", "CrossEncoder"),
]

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def load_log(ds, fname):
    p = RESULTS / ds / "logs" / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    rows = []          # 每行: dict(ds, method, best_epoch, acc, f1, thr, time_s)
    epochs_data = {}   # (ds, method) -> [(epoch, f1), ...]
    for ds in DATASETS:
        for fname, mname in METHODS:
            recs = load_log(ds, fname)
            if not recs:
                print(f"[跳过] {ds}/{fname} 不存在")
                continue
            best = max(recs, key=lambda r: r["val_f1"])
            rows.append({
                "ds": ds, "method": mname,
                "epoch": best["epoch"], "acc": best["val_acc"], "f1": best["val_f1"],
                "thr": best.get("threshold", "argmax"),
                "time": sum(r.get("elapsed_s", 0) for r in recs),
            })
            epochs_data[(ds, mname)] = [(r["epoch"], r["val_f1"]) for r in recs]

    # ── 1. 对比表 ──────────────────────────────────────────────
    md = ["# 实验对比表（自动生成）", "",
          "| 数据集 | 方法 | best epoch | val_acc | val_f1 | 阈值 | 训练耗时(min) |",
          "|--------|------|-----------:|--------:|-------:|-----:|--------------:|"]
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["ds"] == ds]
        if not ds_rows:
            continue
        for r in ds_rows:
            md.append(f"| {r['ds']} | {r['method']} | {r['epoch']} | {r['acc']:.4f} | "
                      f"**{r['f1']:.4f}** | {r['thr']} | {r['time']/60:.1f} |")
        # 数据集冠军
        champ = max(ds_rows, key=lambda r: r["f1"])
        md.append(f"| **{ds} 冠军** | **{champ['method']}** | | | **{champ['f1']:.4f}** | | |")
        md.append("")
    table_path = RESULTS / "comparison_table.md"
    table_path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n对比表 → {table_path}")

    # ── 2. 柱状图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    width, colors = 0.25, ["#2196F3", "#4CAF50", "#FF9800"]
    for i, ds in enumerate(DATASETS):
        vals = [next((r["f1"] for r in rows if r["ds"] == ds and r["method"] == m), 0)
                for _, m in METHODS]
        ax.bar([x + i * width for x in range(3)], vals, width=width,
               label=ds, color=colors[i])
        for x, v in zip([x + i * width for x in range(3)], vals):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks([x + width for x in range(3)])
    ax.set_xticklabels([m for _, m in METHODS], rotation=12)
    ax.set_ylabel("val weighted F1 (best epoch)")
    ax.set_ylim(0, 1)
    ax.set_title("三种方法 × 三个数据集")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "f1_comparison.png", dpi=150)
    plt.close(fig)
    print(f"柱状图 → {RESULTS/'figures'/'f1_comparison.png'}")

    # ── 3. F1-epoch 曲线（3 子图） ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for (m, mname), c in zip(METHODS, colors):
            data = epochs_data.get((ds, mname))
            if not data:
                continue
            xs = [e for e, _ in data]
            ys = [f for _, f in data]
            ax.plot(xs, ys, "o-", label=mname, color=c)
        ax.set_title(ds)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val F1")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "f1_epochs.png", dpi=150)
    plt.close(fig)
    print(f"epoch 曲线 → {RESULTS/'figures'/'f1_epochs.png'}")


if __name__ == "__main__":
    main()
