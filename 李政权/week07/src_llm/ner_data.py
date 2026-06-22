"""人民日报 NER（tokens + ner_tags）的文本/标注转换与 Prompt。"""

import json

ENTITY_TYPES = ["PER", "ORG", "LOC"]

ENTITY_TYPE_ZH = {"PER": "人名", "ORG": "机构名", "LOC": "地名"}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型：PER（人名）、ORG（机构名）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "PER|ORG|LOC"}]}\n'
    '无实体时输出：{"entities": []}'
)

FEW_SHOT_EXAMPLES = [
    {
        "text": "中国人民银行行长易纲在北京发表讲话",
        "output": (
            '{"entities": [{"text": "中国人民银行", "type": "ORG"}, '
            '{"text": "易纲", "type": "PER"}, {"text": "北京", "type": "LOC"}]}'
        ),
    },
    {
        "text": "习近平主席视察了广东省深圳市",
        "output": (
            '{"entities": [{"text": "习近平", "type": "PER"}, '
            '{"text": "广东省", "type": "LOC"}, {"text": "深圳市", "type": "LOC"}]}'
        ),
    },
    {
        "text": "新华社报道了这一消息",
        "output": '{"entities": [{"text": "新华社", "type": "ORG"}]}',
    },
]


def record_to_text(record: dict) -> str:
    return "".join(record["tokens"])


def iter_entities(record: dict):
    """从 BIO 标注迭代 (surface, etype, start, end)。"""
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    i = 0
    while i < len(tokens):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tokens) and ner_tags[i] == f"I-{etype}":
                i += 1
            yield "".join(tokens[start:i]), etype, start, i - 1
        else:
            i += 1


def record_to_target(record: dict) -> str:
    """将一条样本转为 SFT 目标 JSON 字符串。"""
    entities = [{"text": s, "type": t} for s, t, _, _ in iter_entities(record)]
    return json.dumps({"entities": entities}, ensure_ascii=False)


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    return {(s, t, start, end) for s, t, start, end in iter_entities(record)}


def entity_types_in_record(record: dict) -> set[str]:
    return {t for _, t, _, _ in iter_entities(record)}
