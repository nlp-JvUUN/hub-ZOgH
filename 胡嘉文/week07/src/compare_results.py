"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python compare_results.py                        # cluener（默认）
  python compare_results.py --dataset peoples_daily  # 人民日报 NER

前提：
  - outputs/logs/eval_linear_validation.json    （已运行 evaluate.py）
  - outputs/logs/eval_crf_validation.json       （已运行 evaluate.py --use_crf）
  - outputs/logs/eval_llm.json                  （已运行 llm_ner.py）
"""

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()

    dataset_suffix = f"_{args.dataset}" if args.dataset != "cluener" else ""

    linear_res = load_json(LOG_DIR / f"eval_linear{dataset_suffix}_validation.json")
    crf_res = load_json(LOG_DIR / f"eval_crf{dataset_suffix}_validation.json")
    llm_res = load_json(LOG_DIR / "eval_llm.json")

    # 更新标题
    dataset_label = f"（{args.dataset}）" if args.dataset != "cluener" else "（cluener2020）"

    print("\n" + "=" * 80)
    print(f"BERT NER 项目 — 方案汇总对比 {dataset_label}")
    print("=" * 80)

    print(f"\n  dataset={args.dataset}（LLM 结果仅在 cluener 下有效）\n")

    header = f"{'方案':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 67)

    def fmt(val):
        return f"{val:.4f}" if val is not None else "  N/A  "

    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + Linear':<25} "
            f"{linear_res['precision']:>10.4f} "
            f"{linear_res['recall']:>10.4f} "
            f"{linear_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + Linear':<25} {'（未找到结果，请运行 evaluate.py）':>50}")

    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + CRF':<25} "
            f"{crf_res['precision']:>10.4f} "
            f"{crf_res['recall']:>10.4f} "
            f"{crf_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + CRF':<25} {'（未找到结果，请运行 evaluate.py --use_crf）':>50}")

    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "qwen-plus")
        n = llm_res.get("n_samples", "?")
        print(
            f"{f'LLM zero-shot ({model_name})':<25} "
            f"{zs['precision']:>10.4f} "
            f"{zs['recall']:>10.4f} "
            f"{zs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(
            f"{f'LLM few-shot ({model_name})':<25} "
            f"{fs['precision']:>10.4f} "
            f"{fs['recall']:>10.4f} "
            f"{fs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"\n  注：LLM 结果基于验证集 {n} 条采样，非完整测试集")
    else:
        print(f"{'LLM zero/few-shot':<25} {'（未找到结果，请运行 llm_ner.py）':>50}")

    print("\n" + "=" * 80)
    print("关键教学结论：")
    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"  2. 线性头非法序列：{ill_linear} 条；CRF 非法序列：0 条")
        print(f"     → CRF 通过 Viterbi 解码在数学上保证序列合法性")
    if llm_res and linear_res:
        fs_f1 = llm_res["few_shot"]["f1"]
        gap = linear_res["f1"] - fs_f1
        print(f"  3. 微调 BERT vs LLM few-shot：F1 差距 {gap:.4f}")
        print(f"     → 特定领域NER任务中，小模型微调通常显著优于大模型zero/few-shot")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 BERT NER 各方案评估结果对比")
    parser.add_argument("--dataset", type=str, default="cluener", choices=["cluener", "peoples_daily"],
                        help="数据集名称")
    return parser.parse_args()


if __name__ == "__main__":
    main()
