"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集

使用方式：
  python llm_ner.py                                    # CLUENER（默认）
  python llm_ner.py --dataset peoples_daily            # 人民日报
  python llm_ner.py --n_samples 50 --model deepseek-v4-pro

依赖：
  pip install openai
  export DASHSCOPE_API_KEY="sk-262cde904548495c98439e1152d85341"    # DeepSeek API Key（兼容 DASHSCOPE_API_KEY 环境变量）
"""

import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"

# 添加 src/ 到 Python 路径，以便导入 dataset_config
sys.path.insert(0, str(ROOT / "src"))
from dataset_config import get_config, bio_tags_to_spans


def build_client() -> OpenAI:
    # 兼容 DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY 两种环境变量名
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY（你的 DeepSeek API Key）"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def build_system_prompt(entity_types_zh: dict) -> str:
    """根据实体类型动态生成 system prompt。"""
    lines = ["你是一个命名实体识别（NER）专家，专门处理中文文本。"]
    type_list = []
    for etype_en, etype_cn in entity_types_zh.items():
        type_list.append(f"- {etype_en}：{etype_cn}")
    lines.append(f"请从用户输入的文本中识别以下{len(entity_types_zh)}类实体，并以 JSON 格式输出结果：")
    lines.extend(type_list)
    lines.append("")
    lines.append("输出格式（严格遵守，不要包含其他文字）：")
    lines.append('{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}')
    lines.append("")
    lines.append("如果没有实体，输出：{" + '"entities": []}')
    return "\n".join(lines)


# CLUENER 的 few-shot 示例
FEW_SHOT_CLUENER = [
    {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度举了个例子",
        "output": '{"entities": [{"text": "浙商银行", "type": "company"}, {"text": "叶老桂", "type": "name"}]}'
    },
    {
        "text": "《白鹿原》改编自陕西作家陈忠实的同名小说",
        "output": '{"entities": [{"text": "白鹿原", "type": "movie"}, {"text": "陕西", "type": "address"}, {"text": "陈忠实", "type": "name"}, {"text": "白鹿原", "type": "book"}]}'
    },
    {
        "text": "华为技术有限公司总裁任正非在深圳接受了媒体采访",
        "output": '{"entities": [{"text": "华为技术有限公司", "type": "company"}, {"text": "总裁", "type": "position"}, {"text": "任正非", "type": "name"}, {"text": "深圳", "type": "address"}]}'
    },
]

# 人民日报的 few-shot 示例（PER/ORG/LOC）
FEW_SHOT_PEOPLES_DAILY = [
    {
        "text": "中共中央总书记江泽民在北京人民大会堂会见美国客人",
        "output": '{"entities": [{"text": "江泽民", "type": "PER"}, {"text": "北京", "type": "LOC"}, {"text": "美国", "type": "LOC"}]}'
    },
    {
        "text": "中国银行董事长刘明康出席了在香港举行的亚太经合组织会议",
        "output": '{"entities": [{"text": "中国银行", "type": "ORG"}, {"text": "刘明康", "type": "PER"}, {"text": "香港", "type": "LOC"}, {"text": "亚太经合组织", "type": "ORG"}]}'
    },
    {
        "text": "李鹏总理今天在钓鱼台国宾馆会见了日本外相",
        "output": '{"entities": [{"text": "李鹏", "type": "PER"}, {"text": "钓鱼台国宾馆", "type": "LOC"}, {"text": "日本", "type": "LOC"}]}'
    },
]


def build_few_shot_examples(dataset: str) -> list[dict]:
    """按数据集返回对应的 few-shot 示例。"""
    if dataset == "peoples_daily":
        return FEW_SHOT_PEOPLES_DAILY
    return FEW_SHOT_CLUENER


def gold_spans_from_record(record: dict, fmt: str) -> set[tuple[str, str, int, int]]:
    """提取 gold spans，格式：{(text, type, start, end)}。

    支持两种数据格式：
    - "span"：CLUENER {"label": {type: {surface: [[start,end]]}}}
    - "bio"：peoples_daily {"tokens": [...], "ner_tags": ["O", "B-PER", ...]}
    """
    if fmt == "bio":
        return bio_tags_to_spans(record["tokens"], record["ner_tags"])

    spans = set()
    for etype, surfaces in (record.get("label") or {}).items():
        for surface, positions in surfaces.items():
            for start, end in positions:
                spans.add((surface, etype, start, end))
    return spans


def pred_spans_from_response(text: str, response_text: str, valid_types: list[str]) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析 span，格式：{(surface, type, start, end)}。"""
    # 提取 JSON 块（兼容带 markdown 代码块的输出）
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in valid_types:
            continue
        # 在原文中查找位置（取第一次出现）
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


def zero_shot_prompt(text: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str, system_prompt: str, examples: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """调用 LLM API，返回文本输出，带简单重试。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, data_dir: str, entity_types: list[str], seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。"""
    data_path = Path(data_dir) / "validation.json"
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    # 按实体类型分层采样
    by_type = defaultdict(list)
    for r in records:
        # 获取这条记录包含的实体类型
        if isinstance(r.get("label"), dict):
            # CLUENER span 格式
            for etype in r["label"]:
                by_type[etype].append(r)
        elif "ner_tags" in r:
            # peoples_daily BIO 格式
            seen = set()
            for tag in r["ner_tags"]:
                if tag.startswith("B-"):
                    etype = tag[2:]
                    if etype not in seen:
                        by_type[etype].append(r)
                        seen.add(etype)

    selected = set()
    selected_list = []

    # 每类先取 n // len(entity_types) 条
    per_type = max(1, n // len(entity_types))
    for etype in entity_types:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        if candidates:
            chosen = random.sample(candidates, min(per_type, len(candidates)))
            for r in chosen:
                if len(selected_list) < n and id(r) not in selected:
                    selected.add(id(r))
                    selected_list.append(r)

    # 补足到 n 条
    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()

    cfg = get_config(args.dataset)
    entity_types = cfg["entity_types"]
    entity_types_zh = cfg["entity_types_zh"]
    fmt = cfg["format"]
    data_dir = cfg["data_dir"]

    system_prompt = build_system_prompt(entity_types_zh)
    few_shot_examples = build_few_shot_examples(args.dataset)

    client = build_client()
    records = sample_records(args.n_samples, data_dir, entity_types)
    print(f"数据集：{args.dataset}（{fmt} 格式，{len(entity_types)} 类实体）")
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        # 获取文本
        if fmt == "bio":
            text = "".join(record["tokens"])
        else:
            text = record["text"]

        gold = gold_spans_from_record(record, fmt)

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text, system_prompt), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp, entity_types)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text, system_prompt, few_shot_examples), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp, entity_types)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（数据集：{args.dataset}，模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    # 确保数值可 JSON 序列化
    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / f"eval_llm_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print(f"\n下一步：python compare_results.py --dataset {args.dataset}")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--dataset", type=str, default="cluener",
                        choices=["cluener", "peoples_daily"],
                        help="数据集名称（默认 cluener）")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="deepseek-v4-flash",
                        help="模型名称；DeepSeek 支持 deepseek-v4-flash / deepseek-v4-pro")
    return parser.parse_args()


if __name__ == "__main__":
    main()
