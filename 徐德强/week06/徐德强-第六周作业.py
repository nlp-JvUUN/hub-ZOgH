"""
汇总所有实验方法的结果，生成对比表格和柱状图。

教学重点：
  1. 多方法对比：BERT(3种pooling) vs LLM(zero-shot / few-shot / LoRA)
  2. 不只是比准确率，还比训练时间、数据需求量
  3. 可视化让对比一目了然

使用方式：
  python 第六周作业.py
  python 第六周作业.py --output_dir ./outputs --fig_dir ./outputs/figures

依赖：
  pip install matplotlib numpy
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False

# ── 中文字体支持 ──────────────────────────────────────────────────────────────
def _find_chinese_font():
    """查找系统中可用的中文字体。"""
    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
else:
    print("[提示] 未找到中文字体，图表中的中文可能显示为方块")


def load_results(output_dir: Path) -> dict:
    """
    尝试加载各方法的结果文件。
    如果有缺失（某实验还没跑），对应字段为 None。
    """
    results = {}

    # ── BERT 三种 pooling ─────────────────────────────────────────────────────
    for pool in ["cls", "max", "mean"]:
        key = f"BERT-{pool.upper()}"
        log_path = output_dir / f"train_log_{pool}.json"
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                logs = json.load(f)
            best = max(logs, key=lambda x: x.get("val_acc", 0))
            results[key] = {
                "accuracy": best.get("val_acc", None),
                "macro_f1": best.get("val_macro_f1", None),
                "train_time": sum(item.get("elapsed_s", 0) for item in logs),
            }
        else:
            results[key] = None

    # ── Zero-shot ─────────────────────────────────────────────────────────────
    zs_path = output_dir / "llm_zero_shot_results.json"
    if zs_path.exists():
        with open(zs_path, encoding="utf-8") as f:
            zs = json.load(f)
        results["Zero-shot"] = {
            "accuracy": zs.get("accuracy"),
            "train_time": 0.0,
        }
    else:
        results["Zero-shot"] = None

    # ── Few-shot ──────────────────────────────────────────────────────────────
    fs_path = output_dir / "llm_fewshot_results.json"
    if fs_path.exists():
        with open(fs_path, encoding="utf-8") as f:
            fs = json.load(f)
        results[f"Few-shot ({fs.get('n_shot', '?')}-shot)"] = {
            "accuracy": fs.get("accuracy"),
            "train_time": 0.0,
        }
    else:
        results["Few-shot (?-shot)"] = None

    # ── LoRA SFT ──────────────────────────────────────────────────────────────
    sft_log_path = output_dir / "train_log_sft.json"
    sft_res_path = output_dir / "llm_sft_results.json"
    if sft_res_path.exists():
        with open(sft_res_path, encoding="utf-8") as f:
            sft = json.load(f)
        train_time = 0.0
        if sft_log_path.exists():
            with open(sft_log_path, encoding="utf-8") as f:
                sft_logs = json.load(f)
            train_time = sum(item.get("elapsed_s", 0) for item in sft_logs)
        results["LoRA 微调"] = {
            "accuracy": sft.get("accuracy"),
            "train_time": train_time,
        }
    else:
        results["LoRA 微调"] = None

    return results


def print_table(results: dict):
    """打印 Markdown 格式的对比表格。"""
    print("\n" + "=" * 80)
    print("【实验结果汇总】")
    print("=" * 80)

    # 表头
    header = f"{'方法':<25} {'Accuracy':>10} {'Macro F1':>10} {'训练用时':>10} {'训练数据':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    # 方法 → 描述映射
    descriptions = {
        "BERT-CLS":    ("BERT + [CLS] 池化", "53K 条"),
        "BERT-MAX":    ("BERT + Max 池化",   "53K 条"),
        "BERT-MEAN":   ("BERT + Mean 池化",  "53K 条"),
        "Zero-shot":   ("Qwen2.5-0.5B 零样本", "0 条"),
        "Few-shot (5-shot)": ("Qwen2.5-0.5B 5样本示例", "75 条示例"),
        "LoRA 微调":   ("Qwen2.5-0.5B LoRA", "5000 条"),
    }

    for name, info in results.items():
        desc, data_size = descriptions.get(name, ("", "?"))
        row_name = f"{name} ({desc})"

        if info is None:
            print(f"{row_name:<25} {'(未运行)':>10} {'-':>10} {'-':>10} {'-':>12}")
            continue

        acc_str    = f"{info['accuracy']:.4f}" if info.get("accuracy") is not None else "N/A"
        mf1_str    = f"{info.get('macro_f1', 0):.4f}" if info.get("macro_f1") is not None else "N/A"
        time_str   = f"{info.get('train_time', 0):.0f}s" if info.get("train_time") is not None else "N/A"

        print(f"{row_name:<25} {acc_str:>10} {mf1_str:>10} {time_str:>10} {data_size:>12}")

    print("-" * len(header))
    print("\n注：Macro F1 仅 BERT 组有记录（LLM 组按 200 条采样评估，样本少不算 Macro F1）")


def plot_comparison(results: dict, fig_dir: Path):
    """绘制准确率对比柱状图。"""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 过滤掉未运行的实验
    valid = {k: v for k, v in results.items() if v is not None and v.get("accuracy") is not None}
    if not valid:
        print("\n[跳过绘图] 没有可用的实验结果")
        return

    names    = list(valid.keys())
    accs     = [valid[n]["accuracy"] for n in names]
    # 缩短标签用于图表显示
    short_names = [n.replace(" (", "\n(").replace("Few-shot", "Few-shot") for n in names]

    colors = [
        "#4C72B0" if "BERT" in n else "#C44E52" if "LoRA" in n else "#55A868"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(short_names, accs, color=colors, edgecolor="white", linewidth=1.2,
                  alpha=0.85)

    # 在柱子上标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(accs) * 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("文本分类方法 Accuracy 对比（TNEWS 15 分类）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="BERT 组（判别式）"),
        Patch(facecolor="#55A868", label="LLM Prompt 组（不训练）"),
        Patch(facecolor="#C44E52", label="LLM LoRA 微调（生成式）"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    save_path = fig_dir / "methods_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存 → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="汇总所有实验方法结果")
    parser.add_argument("--output_dir", default="../outputs", type=str,
                        help="结果 JSON 文件所在目录")
    parser.add_argument("--fig_dir",    default="../outputs/figures", type=str,
                        help="图表输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    fig_dir    = Path(args.fig_dir).resolve()

    if not output_dir.exists():
        print(f"[错误] 输出目录不存在：{output_dir}")
        print("请先在项目目录下运行各实验步骤")
        sys.exit(1)

    # ── 加载并展示结果 ─────────────────────────────────────────────────────────
    results = load_results(output_dir)
    print_table(results)

    # 统计完成情况
    completed = sum(1 for v in results.values() if v is not None)
    total     = len(results)
    print(f"\n实验完成进度: {completed}/{total}")

    if completed == 0:
        print("\n提示：请按顺序运行以下实验：")
        print("  1. python src/train.py --pool cls  --max_length 64 --batch_size 32 --epochs 3")
        print("  2. python src/train.py --pool max  --max_length 64 --batch_size 32 --epochs 3")
        print("  3. python src/train.py --pool mean --max_length 64 --batch_size 32 --epochs 3")
        print("  4. python src_llm/classify_llm.py --model_path ./Qwen2.5-0.5B-Instruct --num_samples 200")
        print("  5. python classify_fewshot.py --model_path ./Qwen2.5-0.5B-Instruct --num_samples 200 --n_shot 5")
        print("  6. python src_llm/train_sft.py --num_train 5000 --epochs 3 --batch_size 4 --grad_accum 4 --model_path ./Qwen2.5-0.5B-Instruct")
        print("  7. python src_llm/evaluate_sft.py --num_samples 200 --model_path ./Qwen2.5-0.5B-Instruct")
        return

    # ── 绘图 ───────────────────────────────────────────────────────────────────
    if completed >= 2:
        plot_comparison(results, fig_dir)

    print(f"\n所有结果汇总完成。")


if __name__ == "__main__":
    main()
