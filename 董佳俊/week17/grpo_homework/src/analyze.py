"""
analyze.py — 训练前后对比表 + 训练曲线图（matplotlib 不可用时自动降级为文本表）

教学重点：
  1. 同一评估集（相同 seed）配对比较，排除题目差异干扰。
  2. 训练集内难度 vs 未训练难度的泛化差异。
  3. 训练曲线解读：格式分先收敛、正确分后爬坡的典型 RL 动态。

输出：
  outputs/figures/curves.png    训练曲线（若 matplotlib 可用）
  stdout                      格式率 / greedy 正确率 / pass@8 对比表
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

LEVELS = [
    "L1_add_1digit", "L2_addsub_2digit", "L3_addsub_3digit",
    "L4_mul_1digit", "L5_mul_2x1digit", "L6_mul_2x2digit",
]
TRAINED = {"L2_addsub_2digit", "L3_addsub_3digit", "L5_mul_2x1digit"}


def fmt_table(reports):
    base = reports[0][1]
    header = f"{'难度':<20}{'训':^4}" + "".join(f"{name:^34}" for name, _ in reports)
    lines = [header]
    for lv in LEVELS:
        row = f"{lv:<20}{'√' if lv in TRAINED else '—':^4}"
        for _, rep in reports:
            r = rep[lv]
            row += f"{r['greedy_format_rate']:.2f}/{r['greedy_loose_acc']:.2f}/{r['loose_pass@8']:.2f}".center(34)
        lines.append(row)
    return "\n".join(lines)


def plot_curves(log_paths, fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, p in log_paths:
        data = json.load(open(p, encoding="utf-8"))
        steps = [d["step"] for d in data]
        axes[0].plot(steps, [d["loss"] for d in data], label=name)
        axes[1].plot(steps, [d["reward_mean"] for d in data], label=name)
        axes[2].plot(steps, [d["entropy"] for d in data], label=name)
    axes[0].set_title("GRPO loss"); axes[0].set_xlabel("step"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title("Reward (group mean)"); axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].set_title("Policy entropy"); axes[2].set_xlabel("step"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"训练曲线已保存：{fig_path}")


def main():
    base = json.load(open(OUT / "baseline_probe.json", encoding="utf-8"))
    post = json.load(open(OUT / "post_probe.json", encoding="utf-8")) if (OUT / "post_probe.json").exists() else None

    reports = [("基线", base)]
    if post is not None:
        reports.append(("训练后", post))

    print("=" * 100)
    print("训练前后对比（同一评估集；格式率 / greedy正确率 / pass@8）")
    print("=" * 100)
    print(fmt_table(reports))

    # 训练曲线
    log_paths = [("train", OUT / "train_log.json")]
    if (OUT / "train_log.json").exists():
        try:
            plot_curves(log_paths, OUT / "figures" / "curves.png")
        except Exception as e:  # matplotlib 不可用等
            print(f"[analyze] 绘图失败，降级为文本表：{e}")

    # 样例对照（训练集内难度）
    if post is not None:
        print("\n" + "=" * 100)
        print("样例对照（greedy 解码，基线 vs 训练后）")
        for lv in ["L2_addsub_2digit", "L3_addsub_3digit", "L5_mul_2x1digit"]:
            print(f"\n--- {lv} ---")
            for eb, ep in zip(base[lv]["examples"][:3], post[lv]["examples"][:3]):
                print(f"  {eb['expr']} = {eb['answer']}")
                print(f"    前: {eb['greedy_output']!r}")
                print(f"    后: {ep['greedy_output']!r}")


if __name__ == "__main__":
    main()
