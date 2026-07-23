"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python compare_results.py                        # CLUENER（默认）
  python compare_results.py --dataset peoples_daily  # 人民日报

前提：
  - outputs/logs/eval_{dataset}_linear_validation.json   （已运行 evaluate.py）
  - outputs/logs/eval_{dataset}_crf_validation.json      （已运行 evaluate.py --use_crf）
  - outputs/logs/eval_llm_{dataset}.json                 （已运行 llm_ner.py）
"""

import argparse
import json
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
    dataset = args.dataset

    # CLUE cluener2020 test 集标签未公开，使用 validation 集对比
    linear_res = load_json(LOG_DIR / f"eval_{dataset}_linear_validation.json")
    crf_res = load_json(LOG_DIR / f"eval_{dataset}_crf_validation.json")
    llm_res = load_json(LOG_DIR / f"eval_llm_{dataset}.json")
    sft_res = load_json(LOG_DIR / f"eval_sft_{dataset}.json")

    print("\n" + "=" * 80)
    print(f"BERT NER 项目 — 四方案汇总对比（{dataset}）")
    print("=" * 80)

    header = f"{'方案':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 72)

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
        print(f"{'BERT + Linear':<30} {'（未找到结果）':>44}")

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
        print(f"{'BERT + CRF':<30} {'（未找到结果）':>44}")

    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "qwen-plus")
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
        print(f"{'LLM zero/few-shot':<30} {'（未找到结果）':>44}")

    if sft_res:
        m = sft_res.get("metrics", sft_res)
        n_sft = sft_res.get("n_samples", "?")
        print(
            f"{'Qwen2-0.5B SFT (LoRA)':<30} "
            f"{m.get('precision',0):>10.4f} "
            f"{m.get('recall',0):>10.4f} "
            f"{m.get('f1',0):>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"  注：SFT 结果基于验证集 {n_sft} 条采样")

    # 缺失文件提示
    missing = []
    if not linear_res:
        missing.append(f"  python evaluate.py --dataset {dataset}")
    if not crf_res:
        missing.append(f"  python evaluate.py --dataset {dataset} --use_crf")
    if not llm_res:
        missing.append(f"  python src_llm/llm_ner.py --dataset {dataset}")
    if not sft_res:
        missing.append(f"  python src_llm/evaluate_sft.py --dataset {dataset}")

    if missing:
        print(f"\n以下 {dataset} 评估结果尚未生成：")
        for m in missing:
            print(m)

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
    if sft_res and llm_res:
        sft_f1 = sft_res.get("metrics", sft_res).get("f1", 0)
        fs_f1 = llm_res["few_shot"]["f1"]
        print(f"  4. SFT 本地小模型 vs LLM API few-shot：{sft_f1:.4f} vs {fs_f1:.4f}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 NER 多方对比结果")
    parser.add_argument("--dataset", type=str, default="cluener",
                        choices=["cluener", "peoples_daily"],
                        help="数据集名称（默认 cluener）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
