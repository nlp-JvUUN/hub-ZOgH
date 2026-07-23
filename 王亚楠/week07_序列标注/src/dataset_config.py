"""
数据集配置注册中心

集中管理各数据集的元信息（实体类型、目录、格式等），
通过 --dataset 参数即可切换，无需修改代码。

使用方式：
  from dataset_config import get_config, build_label_schema, bio_tags_to_spans
  cfg = get_config("peoples_daily")
  labels, label2id, id2label = build_label_schema(cfg["entity_types"])
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

DATASET_CONFIGS = {
    "cluener": {
        "data_dir": str(ROOT / "data" / "cluener"),
        "entity_types": [
            "address", "book", "company", "game", "government",
            "movie", "name", "organization", "position", "scene",
        ],
        "entity_types_zh": {
            "address": "地址",
            "book": "书名",
            "company": "公司",
            "game": "游戏",
            "government": "政府机构",
            "movie": "影视作品",
            "name": "人名",
            "organization": "组织机构",
            "position": "职位",
            "scene": "景点/场所",
        },
        "format": "span",  # span-based annotation: {"label": {type: {surface: [[start,end]]}}}
        "max_length_default": 128,
    },
    "peoples_daily": {
        "data_dir": str(ROOT / "data" / "peoples_daily"),
        "entity_types": ["PER", "ORG", "LOC"],
        "entity_types_zh": {
            "PER": "人名",
            "ORG": "组织机构",
            "LOC": "地点",
        },
        "format": "bio",   # BIO token-tag format: {"tokens": [...], "ner_tags": ["O", "B-PER", ...]}
        "max_length_default": 256,  # 人民日报句子较长
    },
}


def get_config(dataset_name: str) -> dict:
    """获取数据集配置。

    Args:
        dataset_name: "cluener" 或 "peoples_daily"

    Returns:
        配置字典，包含 data_dir, entity_types, entity_types_zh, format, max_length_default
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"未知数据集 '{dataset_name}'。可选：{list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def build_label_schema(entity_types: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从实体类型列表构建 BIO 标签体系。

    Args:
        entity_types: 实体类型名称列表，如 ["PER", "ORG", "LOC"]

    Returns:
        (labels, label2id, id2label)
        - labels: BIO 标签列表，["O", "B-PER", "I-PER", ...]
        - label2id: 标签名 → id 的映射
        - id2label: id → 标签名 的映射
    """
    labels = ["O"]
    for etype in entity_types:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def bio_tags_to_spans(tokens: list[str], ner_tags: list[str]) -> set[tuple[str, str, int, int]]:
    """从 BIO token-tag 列表中提取实体 span 集合。

    用于 LLM pipeline 中从 peoples_daily 格式提取 gold spans，
    与 CLUENER 的 gold_spans_from_record() 输出格式一致。

    Args:
        tokens: 字符/词 token 列表
        ner_tags: BIO 标签列表，与 tokens 等长

    Returns:
        span 集合，每个元素为 (surface, entity_type, char_start, char_end)
    """
    spans = set()
    text = "".join(tokens)
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                i += 1
            end = i - 1
            # 将 token 索引转换为字符索引
            char_start = len("".join(tokens[:start]))
            char_end = len("".join(tokens[:end + 1])) - 1
            surface = "".join(tokens[start:end + 1])
            spans.add((surface, etype, char_start, char_end))
        else:
            i += 1
    return spans


def bio_tags_to_entities(tokens: list[str], ner_tags: list[str]) -> list[dict]:
    """从 BIO token-tag 列表中提取实体列表。

    用于 SFT 训练中将 peoples_daily 记录转为 target JSON。

    Args:
        tokens: 字符 token 列表
        ner_tags: BIO 标签列表

    Returns:
        实体列表，每个元素为 {"text": str, "type": str}
    """
    entities = []
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                i += 1
            surface = "".join(tokens[start:i])
            entities.append({"text": surface, "type": etype})
        else:
            i += 1
    return entities
