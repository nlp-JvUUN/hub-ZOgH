"""
LLM API 文本匹配对比（DeepSeek）— Zero-shot + Few-shot

教学重点：
  1. Zero-shot vs Few-shot：LLM 在给定少量标注示例后能否大幅提升
  2. Prompt 工程：示例选择、格式设计直接影响 few-shot 效果
  3. 与 Fine-tuned BERT 对比：即使 few-shot 也不及微调，但零成本

使用方式：
  python llm_compare.py                          # 默认 zero-shot 200 条
  python llm_compare.py --shot both               # 跑 zero-shot + few-shot 对比
  python llm_compare.py --shot few --num_samples 100

依赖：
  pip install openai
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.pop("SSL_CERT_FILE", None)
random.seed(42)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT.parent / "data" / "lcqmc"

DEEPSEEK_URL     = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_MODEL   = "deepseek-v4-pro"

# ── Prompt 模板 ─────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """请判断以下两个问题是否表达相同的意思。只回答"是"或"否"，不要有任何其他内容。

问题1：{s1}
问题2：{s2}

回答："""

FEW_SHOT_PROMPT = """请判断以下两个问题是否表达相同的意思。只回答"是"或"否"。

以下是示例（附正确答案）：

{examples}

现在请判断：
问题1：{s1}
问题2：{s2}

回答："""


def build_fewshot_prompt(s1, s2, examples):
    """用标注过的 train 样本构造 few-shot prompt"""
    lines = []
    for i, ex in enumerate(examples, 1):
        label = "是" if ex["label"] == 1 else "否"
        lines.append(f"示例{i}：")
        lines.append(f"  问题1：{ex['sentence1']}")
        lines.append(f"  问题2：{ex['sentence2']}")
        lines.append(f"  回答：{label}")
        lines.append("")
    return FEW_SHOT_PROMPT.format(examples="\n".join(lines), s1=s1, s2=s2)


def load_fewshot_examples(data_dir, n_pos=3, n_neg=3):
    """从训练集抽取 few-shot 示例（与评估集不重叠）"""
    train_path = Path(data_dir) / "train.jsonl"
    pos, neg = [], []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["label"] == 1 and len(pos) < n_pos:
                pos.append(r)
            elif r["label"] == 0 and len(neg) < n_neg:
                neg.append(r)
            if len(pos) >= n_pos and len(neg) >= n_neg:
                break
    examples = pos + neg
    random.shuffle(examples)
    return examples


# ── 单次 LLM 调用 ─────────────────────────────────────────────────────────

def call_llm(client, s1, s2, model, shot="zero", examples=None):
    if shot == "few" and examples:
        prompt = build_fewshot_prompt(s1, s2, examples)
    else:
        prompt = ZERO_SHOT_PROMPT.format(s1=s1, s2=s2)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        finish_reason = resp.choices[0].finish_reason
        # 空返回时重试一次（推理模型思考链过长耗完 token）
        if not answer and finish_reason == "length":
            print(f"  [重试] 首次返回空（思考链过长），max_tokens=512 重试...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip()
        # 首次调用时打印完整 response 用于调试
        if not hasattr(call_llm, "_debug_printed"):
            call_llm._debug_printed = True
            print(f"  [DEBUG] shot={shot}  model={model}")
            print(f"  [DEBUG] finish_reason={resp.choices[0].finish_reason}")
            msg = resp.choices[0].message
            print(f"  [DEBUG] content={msg.content!r}")
            print(f"  [DEBUG] reasoning_content={getattr(msg, 'reasoning_content', 'N/A')!r}")
        if "是" in answer:
            return 1, answer
        elif "否" in answer:
            return 0, answer
        else:
            return -1, answer
    except Exception as e:
        return -1, str(e)


# ── 批量评估 ─────────────────────────────────────────────────────────────

def evaluate_llm(samples, client, model, shot, examples, sleep_sec=0.2):
    results = []
    parse_fail = 0

    for i, r in enumerate(samples):
        pred, raw = call_llm(client, r["sentence1"], r["sentence2"],
                             model, shot=shot, examples=examples)
        if pred == -1:
            parse_fail += 1
            if parse_fail <= 3:
                print(f"  [解析失败 #{parse_fail}] shot={shot} raw={raw!r}")
        results.append({
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "label": r["label"],
            "pred":  pred,
            "raw":   raw,
        })
        if (i + 1) % 10 == 0:
            done = [x for x in results if x["pred"] != -1]
            if done:
                acc = sum(1 for x in done if x["pred"] == x["label"]) / len(done)
                print(f"  [{i+1}/{len(samples)}] {shot}-shot 当前准确率: {acc:.3f}  "
                      f"解析失败: {parse_fail}")
        time.sleep(sleep_sec)

    return results, parse_fail


# ── 并行批量评估 ─────────────────────────────────────────────────────────

def evaluate_llm_parallel(samples, client, model, shot, examples, max_workers=10):
    """多线程并行调用 LLM API，大幅加速"""
    results = [None] * len(samples)
    parse_fail = 0

    def _call(i, r):
        pred, raw = call_llm(client, r["sentence1"], r["sentence2"],
                             model, shot=shot, examples=examples)
        return i, {"sentence1": r["sentence1"], "sentence2": r["sentence2"],
                    "label": r["label"], "pred": pred, "raw": raw}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, i, r) for i, r in enumerate(samples)]
        for j, future in enumerate(as_completed(futures)):
            i, result = future.result()
            results[i] = result
            if result["pred"] == -1:
                parse_fail += 1
            if (j + 1) % 50 == 0:
                done = [r for r in results if r is not None and r["pred"] != -1]
                if done:
                    acc = sum(1 for r in done if r["pred"] == r["label"]) / len(done)
                    print(f"  [{j+1}/{len(samples)}] {shot}-shot 当前准确率: {acc:.3f}  "
                          f"解析失败: {parse_fail}")

    return results, parse_fail


# ── 统计指标 ─────────────────────────────────────────────────────────────

def compute_metrics(results):
    valid = [r for r in results if r["pred"] != -1]
    if not valid:
        return {
            "accuracy": 0.0, "precision_pos": 0.0, "recall_pos": 0.0,
            "f1_pos": 0.0, "n_valid": 0, "n_fail": len(results),
        }

    labels = [r["label"] for r in valid]
    preds  = [r["pred"]  for r in valid]

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    acc  = sum(1 for l, p in zip(labels, preds) if l == p) / len(valid)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "accuracy": acc, "precision_pos": prec, "recall_pos": rec,
        "f1_pos": f1, "n_valid": len(valid),
        "n_fail": len(results) - len(valid),
    }


# ── 打印结果 ─────────────────────────────────────────────────────────────

def print_metrics(metrics, shot_label, n_samples):
    print(f"\n{'='*55}")
    print(f"LLM 评估结果 — {shot_label}（{n_samples} 条样本）")
    print(f"  准确率 (Accuracy)  : {metrics['accuracy']:.4f}")
    print(f"  正例精确率         : {metrics['precision_pos']:.4f}")
    print(f"  正例召回率         : {metrics['recall_pos']:.4f}")
    print(f"  正例 F1            : {metrics['f1_pos']:.4f}")
    print(f"  有效预测数         : {metrics['n_valid']}")
    print(f"  解析失败数         : {metrics['n_fail']}")


def print_comparison(zero_m, few_m):
    print(f"\n{'─'*55}")
    print("Zero-shot vs Few-shot 对比：")
    print(f"  {'指标':<20} {'Zero-shot':>12} {'Few-shot':>12} {'Δ':>10}")
    print(f"  {'-'*54}")
    for key, label in [("accuracy", "Accuracy"), ("precision_pos", "正例精确率"),
                        ("recall_pos", "正例召回率"), ("f1_pos", "正例 F1")]:
        z = zero_m[key]
        f = few_m[key]
        delta = f - z
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {z:>12.4f} {f:>12.4f} {sign}{delta:>9.4f}")


# ── 主流程 ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot / few-shot 文本匹配评估")
    parser.add_argument("--data_dir",    default=str(DATA_DIR), type=str)
    parser.add_argument("--split",       default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--num_samples", default=400, type=int)
    parser.add_argument("--model",       default=DEEPSEEK_MODEL, type=str)
    parser.add_argument("--shot",        default="zero",
                        choices=["zero", "few", "both"],
                        help="zero=零样本, few=少样本, both=两者都跑并对比")
    parser.add_argument("--sleep_sec",   default=0.2, type=float,
                        help="串行模式下的请求间隔（并行模式忽略）")
    parser.add_argument("--parallel",    action="store_true",
                        help="启用多线程并行调用（推荐用于大规模评估）")
    parser.add_argument("--workers",     default=10, type=int,
                        help="并行线程数（默认 10）")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = DEEPSEEK_API_KEY
    if not api_key:
        print("❌ 未设置 DeepSeek API Key")
        return

    # 读取数据
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 随机采样（保持正负比例）
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]
    n_pos = min(args.num_samples // 2, len(pos_rows))
    n_neg = args.num_samples - n_pos
    samples = random.sample(pos_rows, n_pos) + random.sample(neg_rows, min(n_neg, len(neg_rows)))
    random.shuffle(samples)

    print(f"数据集: {data_path.name}  样本数: {len(samples)}")
    print(f"  正样本: {n_pos}  负样本: {len(samples) - n_pos}")
    print(f"模型: {args.model}  模式: {args.shot}")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_URL)

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fewshot_examples = None
    if args.shot in ("few", "both"):
        fewshot_examples = load_fewshot_examples(args.data_dir, n_pos=3, n_neg=3)
        print(f"\nFew-shot 示例（{len(fewshot_examples)} 条）：")
        for ex in fewshot_examples:
            label = "相似" if ex["label"] == 1 else "不相似"
            print(f"  [{label}] {ex['sentence1']!r}  ||  {ex['sentence2']!r}")

    results = {}

    for shot_mode in (["zero", "few"] if args.shot == "both" else [args.shot]):
        # 重置 debug flag
        call_llm._debug_printed = False

        print(f"\n{'#'*55}")
        print(f"# {shot_mode.upper()}-SHOT 评估")
        print(f"{'#'*55}")

        t0 = time.time()
        res, parse_fail = evaluate_llm(samples, client, args.model,
                                       shot_mode, fewshot_examples, args.sleep_sec)
        elapsed = time.time() - t0

        metrics = compute_metrics(res)
        results[shot_mode] = metrics

        shot_label = f"{shot_mode}-shot  (DeepSeek-V4-pro)"
        print_metrics(metrics, shot_label, len(samples))
        print(f"  耗时: {elapsed:.0f}s")

        # 保存结果
        out_path = log_dir / f"llm_{shot_mode}shot_results.json"
        save_data = {
            "model": args.model, "shot": shot_mode,
            "n_samples": len(samples), "metrics": metrics, "results": res,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存 → {out_path}")

        # 打印错误案例
        fail_cases = [r for r in res if r["pred"] != r["label"]][:5]
        if fail_cases:
            print(f"\n  前 5 条预测错误：")
            for r in fail_cases:
                ls = "相似" if r["label"] == 1 else "不相似"
                ps = "相似" if r["pred"] == 1 else ("不相似" if r["pred"] == 0 else "解析失败")
                print(f"    [真:{ls} | 预:{ps}] {r['sentence1']!r}  ||  {r['sentence2']!r}")

    # 对比
    if args.shot == "both":
        print_comparison(results["zero"], results["few"])


if __name__ == "__main__":
    main()
