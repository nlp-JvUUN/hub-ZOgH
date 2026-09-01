# -*- coding: utf-8 -*-
"""
compare.py —— 训练前后对比表 + 训练曲线图
==========================================
输出：
  outputs/figures/train_curves.png    —— GRPO 训练曲线（奖励分量/KL/熵/零方差组）
  outputs/figures/accuracy_compare.png —— 各难度 greedy 正确率 前后对比
  终端打印 markdown 对比表（可直接贴进作业文档）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
for f in ("/System/Library/Fonts/PingFang.ttc",
          "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
    if __import__("os").path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

from data import LEVELS

OUT = "../outputs"


def load_json(p):
    with open(p) as f:
        return json.load(f)


def print_table(base, post):
    print("| 难度 | 在训练集 | 格式率 前→后 | greedy 正确 前→后 | pass@8 前→后 | informative 前→后 |")
    print("|---|---|---|---|---|---|")
    in_train = {"L2": "√", "L1": "—", "L3": "—", "L4": "—"}
    for lv in LEVELS:
        b, p = base[lv], post[lv]
        print(f"| {lv} | {in_train[lv]} | "
              f"{b['greedy_format']:.2f} → {p['greedy_format']:.2f} | "
              f"{b['greedy_acc']:.2f} → {p['greedy_acc']:.2f} | "
              f"{b['pass@8_acc']:.2f} → {p['pass@8_acc']:.2f} | "
              f"{b['informative_rate']:.2f} → {p['informative_rate']:.2f} |")


def plot_curves(log):
    steps = [e["step"] for e in log]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(steps, [e["reward_correct"] for e in log], label="正确分 (1.0)")
    ax.plot(steps, [e["reward_format"] for e in log], label="格式分 (0.2)")
    ax.plot(steps, [e["reward"] for e in log], ls="--", color="gray", label="总分")
    ax.set_ylabel("组平均奖励"); ax.set_title("奖励分量"); ax.legend(); ax.grid(alpha=.3)

    ax = axes[0, 1]
    ax.plot(steps, [e["frac_zero_std"] for e in log], color="tab:red")
    ax.set_ylabel("比例"); ax.set_title("零方差组比例（退化组）"); ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.plot(steps, [e["kl"] for e in log], color="tab:green")
    ax.set_xlabel("step"); ax.set_ylabel("KL(π‖π_ref)"); ax.set_title("KL 正则量"); ax.grid(alpha=.3)

    ax = axes[1, 1]
    ax.plot(steps, [e["entropy"] for e in log], color="tab:purple")
    ax.set_xlabel("step"); ax.set_ylabel("熵 (nat/token)")
    ax.set_title("策略熵（收敛 = 确定性增强）"); ax.grid(alpha=.3)
    fig.tight_layout()
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/train_curves.png", dpi=130)
    print(f"→ {OUT}/figures/train_curves.png")


def plot_accuracy(base, post):
    lvs = LEVELS
    x = np.arange(len(lvs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, [base[l]["greedy_acc"] for l in lvs], w,
                label="SFT 基线", color="#9ecae1")
    b2 = ax.bar(x + w / 2, [post[l]["greedy_acc"] for l in lvs], w,
                label="GRPO 后", color="#fb6a4a")
    ax.bar_label(b1, fmt="%.2f", fontsize=9)
    ax.bar_label(b2, fmt="%.2f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["L1 两不进位加", "L2 两进位加(训)", "L3 三不进位加", "L4 两×一"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("greedy 正确率")
    ax.set_title("GRPO 前后 greedy 正确率（同一评估集 seed=42）")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/accuracy_compare.png", dpi=130)
    print(f"→ {OUT}/figures/accuracy_compare.png")


def main():
    base = load_json(f"{OUT}/baseline_probe.json")
    post = load_json(f"{OUT}/post_train_probe.json")
    tag = os.environ.get("GRPO_TAG", "")
    tlog = load_json(f"{OUT}/train_log{('_' + tag) if tag else ''}.json")
    print("=" * 60)
    print_table(base, post)
    print("=" * 60)
    plot_curves(tlog["log"])
    plot_accuracy(base, post)


if __name__ == "__main__":
    main()
