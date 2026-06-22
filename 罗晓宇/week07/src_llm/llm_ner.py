"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比（适配 peoples_daily 数据集）

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. 使用 seqeval 计算 entity-level F1
  4. 成本控制：只采样 100 条，不跑完整验证集

数据格式：
  peoples_daily 数据集：{"tokens": [...], "ner_tags": [...]}（BIO 标签）

使用方式：
  python llm_ner.py
  python llm_ner.py --n_samples 50 --model qwen-max

依赖：
  pip install openai seqeval
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from seqeval.metrics import f1_score as seqeval_f1

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"

# peoples_daily 数据集的 3 种实体类型
ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地点",
}

ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def pred_bio_from_response(text: str, response_text: str) -> list[str]:
    """从 LLM 输出中解析实体，转换为 BIO 标签序列。"""
    # 初始化全 O
    bio = ["O"] * len(text)
    
    # 提取 JSON 块（兼容带 markdown 代码块的输出）
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return bio

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return bio

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return bio

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES_EN:
            continue
        # 在原文中查找位置（取第一次出现）
        idx = text.find(surface)
        if idx == -1:
            continue
        # 标记 BIO 标签
        bio[idx] = f"B-{etype}"
        for i in range(idx + 1, idx + len(surface)):
            if i < len(bio):
                bio[i] = f"I-{etype}"

    return bio


def compute_seqeval_f1(all_golds: list[list[str]], all_preds: list[list[str]]) -> dict:
    """使用 seqeval 计算 entity-level F1。"""
    from seqeval.metrics import precision_score, recall_score
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = seqeval_f1(all_golds, all_preds)
    return {"precision": p, "recall": r, "f1": f1}


SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文新闻文本。
请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：
- PER：人名（如张三、李四）
- ORG：组织机构（如公司、政府部门、学校等）
- LOC：地点（如城市、国家、景点等）

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}

如果没有实体，输出：{"entities": []}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域",
        "output": '{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    },
    {
        "text": "中国外交部发言人在记者会上表示支持",
        "output": '{"entities": [{"text": "中国外交部", "type": "ORG"}]}'
    },
    {
        "text": "习近平主席在人民大会堂会见外国客人",
        "output": '{"entities": [{"text": "习近平", "type": "PER"}, {"text": "人民大会堂", "type": "LOC"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
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


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。
    
    适配 peoples_daily 格式：{"tokens": [...], "ner_tags": [...]}
    """
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    # 按实体类型分层采样
    by_type = defaultdict(list)
    for r in records:
        # 从 ner_tags 中提取实体类型
        ner_tags = r.get("ner_tags", [])
        etypes_in_record = set()
        for tag in ner_tags:
            if tag.startswith("B-"):
                etypes_in_record.add(tag[2:])
        for etype in etypes_in_record:
            by_type[etype].append(r)

    selected = set()
    selected_list = []

    # 每类先取 n // len(ENTITY_TYPES_EN) 条
    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [r for r in by_type.get(etype, []) if id(r) not in selected]
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

# 提取实体信息用于详情记录
def bio_to_entities(bio: list[str], text: str) -> list[dict]:
    entities = []
    j = 0
    while j < len(bio):
        tag = bio[j]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = j
            while j + 1 < len(bio) and bio[j + 1] == f"I-{etype}":
                j += 1
            end = j
            surface = text[start:end + 1]
            entities.append({"text": surface, "type": etype})
        j += 1
    return entities

def main():
    args = parse_args()

    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        # 将 tokens 拼接为文本
        text = "".join(record.get("tokens", []))
        gold = record.get("ner_tags", [])

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_bio_from_response(text, zs_resp)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_bio_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": bio_to_entities(gold, text),
            "zero_shot": bio_to_entities(zs_pred, text),
            "few_shot": bio_to_entities(fs_pred, text),
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    # 使用 seqeval 计算 entity-level F1
    zs_metrics = compute_seqeval_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_seqeval_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
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

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="qwen-plus")
    return parser.parse_args()


if __name__ == "__main__":
    main()