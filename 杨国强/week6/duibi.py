"""
文本分类不同训练方法效果对比分析脚本

功能:
1. 加载各方法的结果数据
2. 生成性能对比表格
3. 绘制对比可视化图表
4. 输出分析报告

使用方法:
    python compare_methods.py                    # 运行全部分析
    python compare_methods.py --no_plot        # 仅文本输出
    python compare_methods.py --output_dir ./results  # 指定输出目录
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import sys

# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class MethodResult:
    """单个训练方法的实验结果"""
    name: str
    model: str
    trainable_params: str
    train_data: str
    accuracy: float
    macro_f1: float = None
    unparseable_rate: float = 0.0
    training_logs: List[Dict] = None


@dataclass
class ComparisonReport:
    """完整的对比报告数据"""
    methods: List[MethodResult]
    dataset_info: Dict[str, Any]


# ============================================================
# 数据加载
# ============================================================

def load_json(filepath: str) -> Dict:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_results(outputs_dir: Path) -> ComparisonReport:
    """加载所有方法的实验结果"""

    results = []

    # 1. BERT Fine-tune 结果
    bert_log_path = outputs_dir / "train_log_cls.json"
    if bert_log_path.exists():
        logs = load_json(str(bert_log_path))
        last_epoch = logs[-1] if logs else {}
        results.append(MethodResult(
            name="BERT Fine-tune",
            model="bert-base-chinese",
            trainable_params="110M (100%)",
            train_data="53,360 条",
            accuracy=last_epoch.get('val_acc', 0) * 100,
            macro_f1=last_epoch.get('val_macro_f1'),
            training_logs=logs
        ))

    # 2. BERT Fine-tune + 加权 Loss
    bert_weighted_path = outputs_dir / "train_log_cls_weighted.json"
    if bert_weighted_path.exists():
        logs = load_json(str(bert_weighted_path))
        last_epoch = logs[-1] if logs else {}
        results.append(MethodResult(
            name="BERT + 加权 Loss",
            model="bert-base-chinese",
            trainable_params="110M (100%)",
            train_data="53,360 条",
            accuracy=last_epoch.get('val_acc', 0) * 100,
            macro_f1=last_epoch.get('val_macro_f1'),
            training_logs=logs
        ))

    # 3. LLM SFT (LoRA)
    sft_log_path = outputs_dir / "train_log_sft.json"
    sft_result_path = outputs_dir / "llm_sft_results.json"
    if sft_result_path.exists():
        sft_results = load_json(str(sft_result_path))
        logs = load_json(str(sft_log_path)) if sft_log_path.exists() else None
        results.append(MethodResult(
            name="LLM SFT (LoRA)",
            model="Qwen2-0.5B-Instruct",
            trainable_params="1.08M (0.22%)",
            train_data="5,000 条",
            accuracy=sft_results.get('accuracy', 0) * 100,
            training_logs=logs
        ))

    # 4. LLM Zero-shot
    zero_shot_path = outputs_dir / "llm_zero_shot_results.json"
    if zero_shot_path.exists():
        z_results = load_json(str(zero_shot_path))
        total = z_results.get('total', 1)
        unparseable = z_results.get('unparseable', 0)
        results.append(MethodResult(
            name="LLM Zero-shot",
            model="Qwen2-0.5B-Instruct",
            trainable_params="0",
            train_data="0 条",
            accuracy=z_results.get('accuracy', 0) * 100,
            unparseable_rate=unparseable / total * 100 if total > 0 else 0,
            training_logs=None
        ))

    # 数据集信息
    dataset_info = {
        "task": "TNEWS 中文新闻标题分类",
        "num_classes": 15,
        "train_samples": 53360,
        "val_samples": 10000,
        "categories": [
            "故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
            "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"
        ]
    }

    return ComparisonReport(methods=results, dataset_info=dataset_info)


# ============================================================
# 文本输出
# ============================================================

def print_header(title: str, width: int = 80):
    """打印标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table_row(columns: List[str], widths: List[int], separator: str = "|"):
    """打印表格行"""
    row = separator
    for col, width in zip(columns, widths):
        row += f" {col:<{width-2}} {separator}"
    print(row)


def print_comparison_table(methods: List[MethodResult]):
    """打印性能对比表格"""
    print_header("文本分类不同训练方法效果对比")

    # 表格列宽
    widths = [20, 15, 18, 12, 10, 10]

    # 表头
    print_table_row(
        ["方法", "模型", "可训练参数", "训练数据", "准确率", "Macro F1"],
        widths
    )
    print("-" * (sum(widths) + len(widths) * 2 + 1))

    # 数据行
    for m in methods:
        f1_str = f"{m.macro_f1:.3f}" if m.macro_f1 else "-"
        print_table_row(
            [m.name, m.model, m.trainable_params, m.train_data,
             f"{m.accuracy:.2f}%", f1_str],
            widths
        )


def print_detailed_analysis(methods: List[MethodResult]):
    """打印详细分析"""
    print_header("详细分析")

    for m in methods:
        print(f"\n【{m.name}】")
        print(f"  模型: {m.model}")
        print(f"  可训练参数: {m.trainable_params}")
        print(f"  训练数据: {m.train_data}")
        print(f"  准确率: {m.accuracy:.2f}%")

        if m.macro_f1:
            print(f"  Macro F1: {m.macro_f1:.4f}")

        if m.unparseable_rate > 0:
            print(f"  无法解析率: {m.unparseable_rate:.1f}%")

        # 训练日志
        if m.training_logs:
            print("  训练过程:")
            for log in m.training_logs:
                epoch = log.get('epoch', '?')
                train_loss = log.get('train_loss', '-')
                val_acc = log.get('val_acc', '-')
                val_f1 = log.get('val_macro_f1', '-')
                if isinstance(train_loss, float):
                    train_loss = f"{train_loss:.3f}"
                if isinstance(val_acc, float):
                    val_acc = f"{val_acc:.4f}"
                if isinstance(val_f1, float):
                    val_f1 = f"{val_f1:.4f}"
                print(f"    Epoch {epoch}: loss={train_loss}, val_acc={val_acc}, val_f1={val_f1}")


def print_efficiency_comparison(methods: List[MethodResult]):
    """打印效率对比"""
    print_header("效率对比")

    efficiency_data = []
    for m in methods:
        if "BERT" in m.name:
            time_per_epoch = "~12 min"
            inference = "~5ms/条"
        elif "SFT" in m.name:
            time_per_epoch = "~6 min"
            inference = "~60ms/条"
        elif "Zero-shot" in m.name:
            time_per_epoch = "无需训练"
            inference = "~2000ms/条"
        else:
            time_per_epoch = "-"
            inference = "-"

        efficiency_data.append({
            "method": m.name,
            "time_per_epoch": time_per_epoch,
            "inference": inference,
            "trainable": m.trainable_params
        })

    widths = [20, 18, 15, 18]
    print_table_row(["方法", "训练时间/epoch", "推理速度", "可训练参数"], widths)
    print("-" * (sum(widths) + len(widths) * 2 + 1))

    for e in efficiency_data:
        print_table_row(
            [e["method"], e["time_per_epoch"], e["inference"], e["trainable"]],
            widths
        )


def print_conclusions(methods: List[MethodResult]):
    """打印结论与建议"""
    print_header("结论与建议")

    # 找到最优方法
    valid_methods = [m for m in methods if m.unparseable_rate < 5]
    if valid_methods:
        best = max(valid_methods, key=lambda x: x.accuracy)

        print(f"\n【最佳性能】{best.name}")
        print(f"  准确率: {best.accuracy:.2f}%")

        # 找到最高效的方法
        sft_methods = [m for m in methods if "SFT" in m.name]
        if sft_methods:
            sft = sft_methods[0]
            print("\n【最高数据效率】LLM SFT (LoRA)")
            print(f"  仅用 {sft.train_data} 达到 {sft.accuracy:.2f}% 准确率")

    print("\n【场景推荐】")
    print("  * 数据充足、追求精度: BERT Fine-tune + 加权 Loss")
    print("  * 数据有限、资源有限: LLM SFT (LoRA)")
    print("  * 快速原型、无需训练: LLM Zero-shot")
    print("  * 生产环境部署: BERT Fine-tune (推理快、输出可控)")


def print_full_report(report: ComparisonReport):
    """打印完整报告"""
    print_comparison_table(report.methods)
    print_detailed_analysis(report.methods)
    print_efficiency_comparison(report.methods)
    print_conclusions(report.methods)


# ============================================================
# 可视化 (可选)
# ============================================================

def try_import_and_plot(report: ComparisonReport, output_dir: Path):
    """尝试导入 matplotlib 并绘制图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 无头模式

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 准确率对比柱状图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        methods = report.methods
        names = [m.name for m in methods]
        accuracies = [m.accuracy for m in methods]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

        # 左图: 准确率对比
        ax1 = axes[0]
        bars = ax1.bar(names, accuracies, color=colors[:len(names)])
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylim(0, 100)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        # 右图: 资源消耗对比 (示意)
        ax2 = axes[1]
        trainable_counts = [110, 110, 1.08, 0]  # 百万参数
        trainable_labels = ['110M', '110M', '1.08M', '0']

        x_pos = range(len(names))
        ax2.bar(x_pos, trainable_counts, color=colors[:len(names)])
        ax2.set_ylabel('Trainable Parameters (M)')
        ax2.set_title('Parameter Efficiency')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n[图表已保存] {output_dir / 'accuracy_comparison.png'}")

        # 2. 训练过程曲线
        bert_methods = [m for m in methods if "BERT" in m.name and m.training_logs]
        if bert_methods:
            fig2, ax3 = plt.subplots(figsize=(10, 5))

            for m in bert_methods:
                epochs = [log['epoch'] for log in m.training_logs]
                val_accs = [log.get('val_acc', 0) * 100 for log in m.training_logs]
                label = m.name.replace("BERT ", "")
                ax3.plot(epochs, val_accs, marker='o', label=label, linewidth=2)

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Validation Accuracy (%)')
            ax3.set_title('Training Progress Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
            print(f"[图表已保存] {output_dir / 'training_progress.png'}")

        print("[可视化完成] 如需查看，请打开生成的 PNG 文件")

    except ImportError as e:
        print(f"\n[提示] matplotlib 未安装，跳过图表生成")
        print(f"  安装命令: pip install matplotlib")
    except Exception as e:
        print(f"\n[警告] 图表生成失败: {e}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="文本分类方法对比分析")
    parser.add_argument("--outputs_dir", type=str,
                        default=None,
                        help="outputs 目录路径 (默认: 项目根目录下的 outputs)")
    parser.add_argument("--no_plot", action="store_true",
                        help="跳过图表生成")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="图表输出目录")
    args = parser.parse_args()

    # 确定项目根目录和 outputs 目录
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent  # outputs 的上一级是项目根目录

    # 如果 outputs_dir 没有指定，默认为项目根目录下的 outputs
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
        if not outputs_dir.is_absolute():
            outputs_dir = project_dir / outputs_dir
    else:
        outputs_dir = project_dir / "outputs"

    if not outputs_dir.exists():
        print(f"错误: 目录不存在 - {outputs_dir}")
        print("请确保在项目根目录运行，或使用 --outputs_dir 指定正确路径")
        sys.exit(1)

    print(f"加载结果数据: {outputs_dir}")

    # 加载数据
    try:
        report = load_all_results(outputs_dir)
    except Exception as e:
        print(f"错误: 数据加载失败 - {e}")
        sys.exit(1)

    if not report.methods:
        print("错误: 未找到任何实验结果")
        print("请先运行训练脚本生成结果")
        sys.exit(1)

    # 打印报告
    print_full_report(report)

    # 生成图表 (可选)
    if not args.no_plot:
        chart_dir = Path(args.output_dir) if args.output_dir else outputs_dir
        chart_dir.mkdir(parents=True, exist_ok=True)
        try_import_and_plot(report, chart_dir)

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
