"""读取随作业附带的真实结果，输出汇总表并生成训练前后对比图。"""
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LEVELS = [
    "L1_add_1digit", "L2_addsub_2digit", "L3_addsub_3digit",
    "L4_mul_1digit", "L5_mul_2x1digit", "L6_mul_2x2digit",
]


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    baseline = load("baseline_results.json")
    grpo = load("grpo_results.json")
    print(f"{'难度':<22}{'基线正确率':>12}{'GRPO正确率':>14}{'提升':>10}{'GRPO格式率':>14}")
    base_acc, post_acc = [], []
    for level in LEVELS:
        before = baseline[level]["greedy_loose_acc"]
        after = grpo[level]["greedy_loose_acc"]
        base_acc.append(before)
        post_acc.append(after)
        print(f"{level:<22}{before:>12.2%}{after:>14.2%}{after-before:>+10.2%}{grpo[level]['greedy_format_rate']:>14.2%}")
    print("-" * 72)
    print(f"宏平均正确率：{sum(base_acc)/len(base_acc):.2%} -> {sum(post_acc)/len(post_acc):.2%}")

    if plt is None:
        print("未安装 matplotlib，已跳过绘图；执行 pip install matplotlib 后可生成图表。")
        return

    x = range(len(LEVELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], base_acc, width, label="Baseline")
    ax.bar([i + width / 2 for i in x], post_acc, width, label="GRPO")
    ax.set_xticks(list(x), [f"L{i}" for i in range(1, 7)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Greedy accuracy")
    ax.set_title("Arithmetic accuracy before and after GRPO")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUT / "figures" / "accuracy_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"对比图已生成：{path}")


if __name__ == "__main__":
    main()
