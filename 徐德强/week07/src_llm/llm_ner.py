"""
使用 DeepSeek API 做 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集
  5. --dataset 参数：支持 cluener（10类）和 peoples_daily（3类：PER/ORG/LOC）

使用方式：
  python llm_ner.py --dataset cluener
  python llm_ner.py --dataset peoples_daily --n_samples 50

依赖：
  pip install openai
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI

ORIGINAL_DATA = Path(
    "D:/aipy/AI大模型培训部分/week7序列标注问题_0530/"
    "week7 序列标注问题/序列标注项目/data"
)
LOG_DIR = ROOT / "outputs" / "logs"

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# ══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════════════

# cluener：10 类细粒度实体
ENTITY_TYPES_CLUENER = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
]

ENTITY_ZH_CLUENER = {
    "address": "地址", "book": "书名", "company": "公司",
    "game": "游戏", "government": "政府机构", "movie": "影视作品",
    "name": "人名", "organization": "组织机构", "position": "职位",
    "scene": "景点/场所",
}

SYSTEM_PROMPT_CLUENER = (
    "你是一个命名实体识别（NER）专家，专门处理中文文本。\n"
    "请从用户输入的文本中识别以下10类实体，并以 JSON 格式输出结果：\n"
    "- address：地址（如街道、城市）\n"
    "- book：书名\n"
    "- company：公司名称\n"
    "- game：游戏名称\n"
    "- government：政府机构名称\n"
    "- movie：影视作品名称\n"
    "- name：人名\n"
    "- organization：组织机构名称\n"
    "- position：职位名称\n"
    "- scene：景点或场所名称\n\n"
    "输出格式（严格遵守，不要包含其他文字）：\n"
    '{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}\n\n'
    "如果没有实体，输出：{\"entities\": []}"
)

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

# peoples_daily：3 类实体（PER 人名、ORG 组织机构、LOC 地名）
ENTITY_TYPES_PD = ["PER", "ORG", "LOC"]

ENTITY_ZH_PD = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}

SYSTEM_PROMPT_PD = (
    "你是一个命名实体识别（NER）专家，专门处理中文文本。\n"
    "请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：\n"
    "- PER：人名（如张三、李四）\n"
    "- ORG：组织机构名称（如北京大学、国务院）\n"
    "- LOC：地名（如北京、上海市）\n\n"
    "输出格式（严格遵守，不要包含其他文字）：\n"
    '{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}\n\n'
    "如果没有实体，输出：{\"entities\": []}"
)

FEW_SHOT_PD = [
    {
        "text": "邓小平同志在深圳视察时指出，发展才是硬道理",
        "output": '{"entities": [{"text": "邓小平", "type": "PER"}, {"text": "深圳", "type": "LOC"}]}'
    },
    {
        "text": "北京大学和清华大学位于北京市海淀区",
        "output": '{"entities": [{"text": "北京大学", "type": "ORG"}, {"text": "清华大学", "type": "ORG"}, {"text": "北京市", "type": "LOC"}, {"text": "海淀区", "type": "LOC"}]}'
    },
    {
        "text": "中国足协副主席张吉龙出席了在北京举行的新闻发布会",
        "output": '{"entities": [{"text": "中国足协", "type": "ORG"}, {"text": "张吉龙", "type": "PER"}, {"text": "北京", "type": "LOC"}]}'
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# API 客户端
# ══════════════════════════════════════════════════════════════════════════════

def build_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gold span 提取（按数据集格式区分）
# ══════════════════════════════════════════════════════════════════════════════

def gold_spans_from_record_cluener(record: dict) -> set[tuple[str, str, int, int]]:
    """cluener：从 span label 提取 gold spans。"""
    spans = set()
    for etype, surfaces in (record.get("label") or {}).items():
        for surface, positions in surfaces.items():
            for start, end in positions:
                spans.add((surface, etype, start, end))
    return spans


def gold_spans_from_record_pd(record: dict) -> set[tuple[str, str, int, int]]:
    """peoples_daily：从 BIO ner_tags 反解析 gold spans。

    与 src/dataset.py 的 bio_tags_to_entities() 逻辑一致，但直接返回 span 格式。
    """
    from dataset import bio_tags_to_entities

    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    entities = bio_tags_to_entities(tokens, ner_tags)

    spans = set()
    for ent in entities:
        # 在原文中查找位置
        text_str = "".join(tokens)
        idx = text_str.find(ent["text"])
        if idx != -1:
            spans.add((ent["text"], ent["type"], idx, idx + len(ent["text"]) - 1))
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# LLM 输出解析
# ══════════════════════════════════════════════════════════════════════════════

def pred_spans_from_response(text: str, response_text: str,
                             entity_types: list[str]) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析 span，格式：{(surface, type, start, end)}。"""
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
        if not surface or etype not in entity_types:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


# ══════════════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════════════

def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


# ══════════════════════════════════════════════════════════════════════════════
# Prompt 构建
# ══════════════════════════════════════════════════════════════════════════════

def zero_shot_prompt(text: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str, system_prompt: str,
                    examples: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


# ══════════════════════════════════════════════════════════════════════════════
# API 调用
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 采样
# ══════════════════════════════════════════════════════════════════════════════

def sample_records_cluener(data_dir: Path, n: int, seed: int = 42) -> list[dict]:
    """cluener 采样：按实体类型分层采样。"""
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        for etype in (r.get("label") or {}):
            by_type[etype].append(r)

    selected = set()
    selected_list = []
    per_type = max(1, n // len(ENTITY_TYPES_CLUENER))
    for etype in ENTITY_TYPES_CLUENER:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def sample_records_pd(data_dir: Path, n: int, seed: int = 42) -> list[dict]:
    """peoples_daily 采样：按实体类型分层采样。"""
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    # 按 BIO tag 中含有的实体类型分组
    by_type = defaultdict(list)
    for r in records:
        tags = r.get("ner_tags", [])
        types_in_record = set()
        for tag in tags:
            if tag.startswith("B-") or tag.startswith("I-"):
                types_in_record.add(tag[2:])
        for etype in types_in_record:
            by_type[etype].append(r)

    selected = set()
    selected_list = []
    per_type = max(1, n // len(ENTITY_TYPES_PD))
    for etype in ENTITY_TYPES_PD:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    ds = args.dataset

    # 根据数据集选择配置
    if ds == "peoples_daily":
        data_dir = ORIGINAL_DATA / "peoples_daily"
        entity_types = ENTITY_TYPES_PD
        system_prompt = SYSTEM_PROMPT_PD
        few_shot_examples = FEW_SHOT_PD
        gold_spans_fn = gold_spans_from_record_pd
        sample_fn = sample_records_pd
    else:
        data_dir = ORIGINAL_DATA / "cluener"
        entity_types = ENTITY_TYPES_CLUENER
        system_prompt = SYSTEM_PROMPT_CLUENER
        few_shot_examples = FEW_SHOT_CLUENER
        gold_spans_fn = gold_spans_from_record_cluener
        sample_fn = sample_records_cluener

    client = build_client()
    records = sample_fn(data_dir, args.n_samples)
    print(f"数据集：{ds}，采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = record.get("text") or "".join(record["tokens"])
        gold = gold_spans_fn(record)

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
    print(f"LLM NER 对比结果（模型：{args.model}，数据集：{ds}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} "
          f"{zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} "
          f"{fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "dataset": ds,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / f"eval_llm_{ds}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print(f"\n下一步：python compare_results.py --dataset {ds}")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--dataset", type=str, choices=["cluener", "peoples_daily"],
                        default="peoples_daily", help="数据集选择")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default=DEEPSEEK_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    main()
