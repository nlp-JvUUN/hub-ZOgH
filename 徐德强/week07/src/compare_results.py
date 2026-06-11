"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python compare_results.py --dataset cluener
  python compare_results.py --dataset peoples_daily

前提（peoples_daily 为例）：
  - outputs/logs/eval_linear_validation_peoples_daily.json  （已运行 evaluate.py）
  - outputs/logs/eval_crf_validation_peoples_daily.json     （已运行 evaluate.py --use_crf）
  - outputs/logs/eval_llm_peoples_daily.json                （已运行 llm_ner.py）
  - outputs/logs/eval_sft_peoples_daily.json                （已运行 evaluate_sft.py）
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
    ds = args.dataset

    linear_res  = load_json(LOG_DIR / f"eval_linear_validation_{ds}.json")
    crf_res     = load_json(LOG_DIR / f"eval_crf_validation_{ds}.json")
    llm_res     = load_json(LOG_DIR / f"eval_llm_{ds}.json")
    sft_res     = load_json(LOG_DIR / f"eval_sft_{ds}.json")

    print("\n" + "=" * 80)
    print(f"BERT NER 项目 — 四方案汇总对比（数据集：{ds}）")
    print("=" * 80)

    header = f"{'方案':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 72)

    # BERT + Linear
    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + Linear':<30} "
            f"{linear_res['precision']:>10.4f} "
            f"{linear_res['recall']:>10.4f} "
            f"{linear_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + Linear':<30} {'（未找到结果）':>50}")

    # BERT + CRF
    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + CRF':<30} "
            f"{crf_res['precision']:>10.4f} "
            f"{crf_res['recall']:>10.4f} "
            f"{crf_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + CRF':<30} {'（未找到结果）':>50}")

    # LLM API
    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "?")
        n = llm_res.get("n_samples", "?")
        print(
            f"{f'LLM zero-shot ({model_name})':<30} "
            f"{zs['precision']:>10.4f} "
            f"{zs['recall']:>10.4f} "
            f"{zs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(
            f"{f'LLM few-shot ({model_name})':<30} "
            f"{fs['precision']:>10.4f} "
            f"{fs['recall']:>10.4f} "
            f"{fs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"\n  注：LLM 结果基于验证集 {n} 条采样，非完整测试集")
    else:
        print(f"{'LLM zero/few-shot':<30} {'（未找到结果）':>50}")

    # LLM SFT
    if sft_res:
        m = sft_res["metrics"]
        pf = sft_res.get("parse_fail", "?")
        print(
            f"{'LLM SFT (LoRA)':<30} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"  注：SFT 结果基于 {sft_res.get('n_samples', '?')} 条采样，JSON 解析失败 {pf} 条")
    else:
        print(f"{'LLM SFT (LoRA)':<30} {'（未找到结果）':>50}")

    # ── 关键教学结论 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("关键教学结论：")

    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"  2. 线性头非法序列：{ill_linear} 条；CRF 非法序列："
              f"{crf_res['illegal_stats']['total_illegal']} 条")
        print(f"     → CRF 通过 Viterbi 解码在数学上保证序列合法性")

    if llm_res and linear_res:
        fs_f1 = llm_res["few_shot"]["f1"]
        gap = linear_res["f1"] - fs_f1
        print(f"  3. 微调 BERT vs LLM few-shot：F1 差距 {gap:.4f}")
        print(f"     → 特定领域NER任务中，小模型微调通常显著优于大模型zero/few-shot")

    if sft_res and linear_res:
        sft_f1 = sft_res["metrics"]["f1"]
        gap2 = linear_res["f1"] - sft_f1
        print(f"  4. BERT+Linear vs LLM SFT：F1 差距 {gap2:.4f}")
        print(f"     → 序列标注在 NER 精确边界任务上天然优于生成式方法")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 NER 四方案对比结果")
    parser.add_argument("--dataset", type=str, choices=["cluener", "peoples_daily"],
                        default="peoples_daily", help="数据集选择")
    return parser.parse_args()


if __name__ == "__main__":
    main()
