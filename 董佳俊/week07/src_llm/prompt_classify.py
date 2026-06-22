"""
LLM 文本分类：zero-shot vs few-shot 提示词对比

目的：测试"不用训练数据"做文本分类的下限和上限
  - zero-shot：全靠 LLM 预训练知识 + 任务描述
  - few-shot ：在 prompt 中提供 4 个标注示例，引导模型对齐输出格式

LLM 直接做情感分类，不依赖训练数据。对比 zero-shot（纯指令）和 few-shot（给 4 个示例）的效果差异。

用法：
  python src_llm/prompt_classify.py
  python src_llm/prompt_classify.py --n_samples 50 --model deepseek-chat
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"


def build_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ── Prompt 设计 ──

SYSTEM_PROMPT = """你是一个精通中文情感分析（Sentiment Analysis）的专家。
请对用户输入的文本进行情感倾向判断，只输出 JSON 格式结果（不要包含其他文字）：

输出格式（严格遵守）：
{"sentiment": "正面", "confidence": 0.95}

规则：
  - sentiment 取值必须是 "正面" 或 "负面"
  - confidence 取值 0~1 之间，表示你的置信度
  - 如果文本是中性的，默认归为"正面"
  - 没有情感倾向时，自信度应低于 0.6"""

# 4 个 few-shot 示例（2 正 2 负，覆盖不同类型表达）
FEW_SHOT_EXAMPLES = [
    {
        "text": "这家店的服务太差了，等了一个小时才上菜，以后再也不会来了。",
        "output": '{"sentiment": "负面", "confidence": 0.98}'
    },
    {
        "text": "酒店位置很好，出门就是地铁，房间也很干净，推荐！",
        "output": '{"sentiment": "正面", "confidence": 0.95}'
    },
    {
        "text": "物流速度还行，但包装破损了，里面的东西磕掉了一块漆。",
        "output": '{"sentiment": "负面", "confidence": 0.85}'
    },
    {
        "text": "客服态度特别好，帮我解决了问题，还主动给了优惠券，非常满意。",
        "output": '{"sentiment": "正面", "confidence": 0.97}'
    },
]


def zero_shot_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_response(resp: str) -> dict | None:
    """从 LLM 输出中提取 JSON 并解析情感标签。"""
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    sentiment = str(obj.get("sentiment", "")).strip()
    confidence = float(obj.get("confidence", 0.5))
    if sentiment not in ("正面", "负面"):
        return None
    return {"label": 1 if sentiment == "正面" else 0, "confidence": confidence}


def call_api(client, messages: list[dict], model: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=256,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    API 错误: {e}")
                return ""
    return ""


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
    }


def main():
    args = parse_args()

    import sys; sys.path.insert(0, str(ROOT))
    from data_loader import load_dataset
    records = load_dataset("test")
    random.seed(42)
    if args.n_samples < len(records):
        records = random.sample(records, args.n_samples)
    print(f"测试样本: {len(records)} 条（从 test 集采样）")

    client = build_client()

    y_true = [r["label"] for r in records]
    zs_preds, fs_preds = [], []
    details = []

    for i, r in enumerate(records):
        text = r["text"]
        gold = r["label"]

        zs_raw = call_api(client, zero_shot_messages(text), args.model)
        zs = parse_response(zs_raw)
        fs_raw = call_api(client, few_shot_messages(text), args.model)
        fs = parse_response(fs_raw)

        zs_label = zs["label"] if zs else -1
        fs_label = fs["label"] if fs else -1
        zs_preds.append(zs_label)
        fs_preds.append(fs_label)

        details.append({
            "text": text,
            "gold": gold,
            "zero_shot": {"raw": zs_raw, "parsed": zs},
            "few_shot": {"raw": fs_raw, "parsed": fs},
        })

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"  进度 {i + 1}/{len(records)}")

    # 过滤解析失败的样本
    valid_zs = [(t, p) for t, p in zip(y_true, zs_preds) if p >= 0]
    valid_fs = [(t, p) for t, p in zip(y_true, fs_preds) if p >= 0]
    zs_parse_ok = len(valid_zs)
    fs_parse_ok = len(valid_fs)

    print(f"\n{'=' * 60}")
    print(f"  LLM 文本分类结果（模型: {args.model}）")
    print(f"{'=' * 60}")
    print(f"  Zero-shot 解析成功: {zs_parse_ok}/{len(records)}")
    print(f"  Few-shot  解析成功: {fs_parse_ok}/{len(records)}")

    zs_metrics = compute_metrics(*zip(*valid_zs)) if valid_zs else {}
    fs_metrics = compute_metrics(*zip(*valid_fs)) if valid_fs else {}

    print(f"\n  {'方法':<25} {'Accuracy':>8} {'F1-macro':>8}")
    print(f"  {'─' * 43}")
    print(f"  {'Zero-shot (无示例)':<25} {zs_metrics.get('accuracy', 0):>8.4f} {zs_metrics.get('f1', 0):>8.4f}")
    print(f"  {'Few-shot (4 示例)':<25} {fs_metrics.get('accuracy', 0):>8.4f} {fs_metrics.get('f1', 0):>8.4f}")

    # 保存
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot_parse_rate": zs_parse_ok / len(records),
        "few_shot_parse_rate": fs_parse_ok / len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "details": details,
    }
    out_path = LOG_DIR / "llm_prompt_cls.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存 → {out_path}")
    print(f"  下一步：python compare_all.py")


def parse_args():
    p = argparse.ArgumentParser(description="LLM zero-shot/few-shot 文本分类")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--model", type=str, default="deepseek-chat")
    return p.parse_args()


if __name__ == "__main__":
    main()
