"""
汇总对比所有文本分类方法的实验效果

对比维度：
  1. 是否需要训练数据（传统 ML、BERT、SFT 需要 / zero-shot、few-shot 不需要）
  2. 训练参数量（传统 ML 零参 vs BERT 全量 100M vs LoRA ~0.3M）
  3. 推理速度（传统 ML 最快 > BERT > LLM API）
  4. 分类效果（accuracy / F1 在测试集上的表现）

方法列表（共 7 种）：
  ┌─────────────────────────────────────────────────────────────┐
  │ 类别              方法                   需要训练?  参数量    │
  ├─────────────────────────────────────────────────────────────┤
  │ 传统 ML           TF-IDF + LR               是       ~0     │
  │                   TF-IDF + SVM              是       ~0     │
  │                   TF-IDF + XGBoost          是       ~0     │
  │ BERT 微调         BERT 全量微调             是      102M    │
  │                   BERT 冻结+分类头          是      0.6M    │
  │                   BERT LoRA 微调            是      0.3M    │
  │ LLM 提示          LLM zero-shot             否       0      │
  │                   LLM few-shot              否       0      │
  │ LLM 微调          Qwen2.5 LoRA SFT          是      0.8M    │
  └─────────────────────────────────────────────────────────────┘

"""

import json
import argparse
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"

# ── 预定义的 9 种方法及其元信息 ──
METHOD_META = OrderedDict({
    "TF-IDF + LR":          {"category": "传统 ML", "needs_train": True,  "params": "~5K"},
    "TF-IDF + SVM":         {"category": "传统 ML", "needs_train": True,  "params": "~5K"},
    "TF-IDF + XGBoost":     {"category": "传统 ML", "needs_train": True,  "params": "~5K"},
    "BERT-full":            {"category": "BERT 微调", "needs_train": True,  "params": "~102M"},
    "BERT-freeze":          {"category": "BERT 微调", "needs_train": True,  "params": "~0.6M"},
    "BERT-lora":            {"category": "BERT 微调", "needs_train": True,  "params": "~0.3M"},
    "LLM zero-shot":        {"category": "LLM 提示",  "needs_train": False, "params": "0"},
    "LLM few-shot":         {"category": "LLM 提示",  "needs_train": False, "params": "0"},
    "Qwen2.5 SFT-LoRA":     {"category": "LLM 微调",  "needs_train": True,  "params": "~0.8M"},
})


def collect_results() -> list[dict]:
    """从日志文件中收集所有已有结果。"""
    results = []

    # 传统 ML
    ml_path = LOG_DIR / "ml_baseline.json"
    if ml_path.exists():
        with open(ml_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                r["_source"] = str(ml_path)
                results.append(r)

    # BERT 微调
    for method in ["full", "freeze", "lora"]:
        bp = LOG_DIR / f"bert_{method}.json"
        if bp.exists():
            with open(bp, "r", encoding="utf-8") as f:
                d = json.load(f)
                d["_source"] = str(bp)
                results.append(d)

    # LLM 提示
    lp = LOG_DIR / "llm_prompt_cls.json"
    if lp.exists():
        with open(lp, "r", encoding="utf-8") as f:
            d = json.load(f)
            # 拆成两条
            if "zero_shot" in d and d["zero_shot"]:
                results.append({
                    "method": "LLM zero-shot",
                    "test_metrics": {
                        "accuracy": d["zero_shot"].get("accuracy", 0),
                        "f1": d["zero_shot"].get("f1", 0),
                    },
                    "_source": str(lp),
                })
            if "few_shot" in d and d["few_shot"]:
                results.append({
                    "method": "LLM few-shot",
                    "test_metrics": {
                        "accuracy": d["few_shot"].get("accuracy", 0),
                        "f1": d["few_shot"].get("f1", 0),
                    },
                    "_source": str(lp),
                })

    # LLM SFT
    for tag in ["lora", "full"]:
        sp = LOG_DIR / f"sft_{tag}_cls.json"
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                d = json.load(f)
                d["_source"] = str(sp)
                results.append(d)

    return results


def normalize_result(r: dict) -> dict:
    """将不同来源的结果统一格式。"""
    method = r.get("method", "unknown")
    metrics = r.get("test_metrics", {})

    # 兼容 ml_baseline 的字段名（没有 test_metrics 包装）
    if not metrics and "accuracy" in r:
        metrics = {
            "accuracy": r.get("accuracy", 0),
            "f1": r.get("f1_macro", r.get("f1", 0)),
        }

    return {
        "method": method,
        "category": METHOD_META.get(method, {}).get("category", "未知"),
        "needs_train": METHOD_META.get(method, {}).get("needs_train", True),
        "params": METHOD_META.get(method, {}).get("params", "?"),
        "accuracy": metrics.get("accuracy", None),
        "f1": metrics.get("f1", None),
        "precision": metrics.get("precision", None),
        "recall": metrics.get("recall", None),
    }


def print_table(results: list[dict]):
    """打印对比表格。"""
    print(f"\n{'=' * 85}")
    print(f"  文本分类 —— 不同训练方法效果对比")
    print(f"{'=' * 85}")

    header = f"  {'方法':<22} {'类别':<12} {'需训练':<8} {'参数量':<10} {'Acc':>8} {'F1':>8}"
    print(header)
    print(f"  {'─' * 83}")

    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "   N/A"
        f1_str = f"{r['f1']:.4f}" if r['f1'] is not None else "   N/A"
        train_str = "是" if r['needs_train'] else "否"
        print(f"  {r['method']:<22} {r['category']:<12} {train_str:<8} {r['params']:<10} {acc_str:>8} {f1_str:>8}")

    print(f"\n  N/A = 尚未运行，请先执行对应脚本")


def plot_comparison(results: list[dict]):
    """生成对比柱状图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("  (matplotlib 未安装，跳过图表生成)")
        return

    # 只画有数据的
    valid = [r for r in results if r["accuracy"] is not None]
    if len(valid) < 2:
        print("  (至少需要 2 种方法有结果才能画图)")
        return

    methods = [r["method"] for r in valid]
    accs = [r["accuracy"] * 100 for r in valid]
    f1s = [r["f1"] * 100 for r in valid]
    categories = [r["category"] for r in valid]

    # 按类别分颜色
    cat_colors = {"传统 ML": "#4C72B0", "BERT 微调": "#55A868", "LLM 提示": "#C44E52", "LLM 微调": "#8172B2"}
    colors = [cat_colors.get(c, "#888888") for c in categories]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(methods)))
    w = 0.35

    bars1 = ax.bar([i - w / 2 for i in x], accs, w, label="Accuracy (%)", color=colors, alpha=0.85, edgecolor="white")
    bars2 = ax.bar([i + w / 2 for i in x], f1s, w, label="F1-macro (%)", color=[c for c in colors], alpha=0.4,
                   edgecolor="white", hatch="//")

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("分数 (%)")
    ax.set_title("文本分类：不同训练方法效果对比", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "method_comparison.png", dpi=150)
    print(f"\n  图表已保存 → {FIG_DIR / 'method_comparison.png'}")
    plt.close()


def main():
    parse_args()

    results_raw = collect_results()
    if not results_raw:
        print("未找到任何实验结果日志。请先运行至少一种方法：")
        print("  python src_traditional/ml_baseline.py")
        print("  python src_bert/bert_trainer.py --method full")
        print("  python src_llm/prompt_classify.py")
        print("  python src_llm/sft_finetune.py")
        return

    results = [normalize_result(r) for r in results_raw]
    print_table(results)

    # 按 category 做小结
    print(f"\n{'─' * 60}")
    print("  要点小结")
    print(f"{'─' * 60}")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in by_cat.items():
        best = max(items, key=lambda x: x["f1"] or 0)
        print(f"  [{cat}] 最佳: {best['method']} (F1={best['f1']:.4f})")

    all_f1 = [(r["f1"] or 0) for r in results]
    valid_f1 = [f for f in all_f1 if f > 0]
    if valid_f1:
        best_all = max(results, key=lambda x: x["f1"] or 0)
        print(f"\n  ★ 全局最佳: {best_all['method']} (F1={best_all['f1']:.4f})")

    # 对比核心发现
    print(f"\n{'─' * 60}")
    print("  核心对比维度")
    print(f"{'─' * 60}")
    print("  1. 传统 ML vs BERT：特征工程 vs 端到端学习的差距")
    print("  2. BERT freeze vs full：预训练知识迁移的价值")
    print("  3. BERT full vs LoRA：全量微调 vs 高效微调的效率 trade-off")
    print("  4. LLM zero-shot vs few-shot：提示工程示例数量的影响")
    print("  5. LLM 提示 vs 微调：上下文学习 vs 参数更新的效果差异")

    plot_comparison(results)


def parse_args():
    p = argparse.ArgumentParser(description="汇总对比所有文本分类方法的实验结果")
    return p.parse_args()


if __name__ == "__main__":
    main()
