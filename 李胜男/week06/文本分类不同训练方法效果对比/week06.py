"""
文本分类不同训练方法效果对比
数据集：CLUE TNEWS (15类)

对比维度：
  1. 池化策略：CLS vs Mean vs Max
  2. 类别不均衡处理：普通 Loss vs 加权 Loss
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(r"E:\八斗\week6 文本分类问题\text_classification项目")
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"

def run_single_experiment(pool: str, use_weight: bool):
    """
    调用 train.py 运行单次实验
    """
    exp_name = f"{pool}_{'weighted' if use_weight else 'normal'}"
    print(f"\n{'#' * 60}")
    print(f"# 开始实验: {exp_name}")
    print(f"# 池化策略: {pool}, 加权Loss: {use_weight}")
    print(f"{'#' * 60}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "train.py"),
        "--pool", pool,
        "--epochs", "3",
        "--batch_size", "32",
        "--lr", "2e-5"
    ]

    if use_weight:
        cmd.append("--use_class_weight")

    try:
        # 运行训练脚本
        subprocess.run(cmd, check=True, cwd=SRC_DIR)
        print(f"✅ 实验 {exp_name} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验 {exp_name} 失败: {e}")
        return False


def collect_results():
    """
    收集所有实验的结果
    """
    results = []

    # 对比的6种情况
    experiments = [
        ("cls", False), ("cls", True),
        ("mean", False), ("mean", True),
        ("max", False), ("max", True),
    ]

    for pool, use_weight in experiments:
        tag = f"{pool}_weighted" if use_weight else pool
        log_path = OUTPUT_DIR / f"train_log_{tag}.json"

        if not log_path.exists():
            print(f"[警告] 找不到日志: {log_path}")
            continue

        with open(log_path, encoding="utf-8") as f:
            logs = json.load(f)

        # 取最后一个 epoch 的结果（或者最佳 val_acc）
        final_log = logs[-1]

        results.append({
            "实验组": tag,
            "池化策略": pool,
            "加权Loss": "是" if use_weight else "否",
            "训练Loss": f"{final_log['train_loss']:.4f}",
            "训练Acc": f"{final_log['train_acc']:.4f}",
            "验证Acc": f"{final_log['val_acc']:.4f}",
            "验证F1": f"{final_log['val_macro_f1']:.4f}",
        })

    return results


def print_comparison_table(results: list):
    """
    打印对比表格
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "不同训练方法效果对比报告")
    print("=" * 80)

    # 表头
    header = f"{'实验组':<20} | {'池化':<5} | {'加权Loss':<8} | {'Val Acc':<8} | {'Macro F1':<8}"
    print(header)
    print("-" * 80)

    # 按 F1 排序，方便看效果
    results.sort(key=lambda x: float(x['验证F1']), reverse=True)

    for r in results:
        row = f"{r['实验组']:<20} | {r['池化策略']:<5} | {r['加权Loss']:<8} | {r['验证Acc']:<8} | {r['验证F1']:<8}"
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("注意：由于使用 CPU，预计总耗时较长，请耐心等待...")

    # 定义实验组合
    experiments = [
        ("cls", False), ("cls", True),
        ("mean", False), ("mean", True),
        ("max", False), ("max", True),
    ]

    # 依次运行实验
    for pool, use_weight in experiments:
        success = run_single_experiment(pool, use_weight)
        if not success:
            print("发生错误，停止后续实验。请检查 train.py 是否正常运行。")
            break

    # 收集并打印结果
    results = collect_results()
    if results:
        print_comparison_table(results)
    else:
        print("未能收集到任何结果，请检查 outputs/train_log_*.json 是否存在。")

    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
