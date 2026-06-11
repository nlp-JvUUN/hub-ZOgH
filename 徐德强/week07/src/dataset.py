"""
NER 数据集加载 & BIO 标签体系构建，支持 cluener 和 peoples_daily 两套数据集

教学重点：
  1. cluener：span 标注格式（label dict）→ 需 span_to_bio() 转换为 BIO
  2. peoples_daily：已是 BIO 格式，直接使用 ner_tags 字段，无需转换
  3. BERT 子词对齐：word_ids() 策略，非首子词标签设为 -100
  4. 标签体系差异：cluener 21 标签（10实体类型）→ peoples_daily 7 标签（3实体类型）

使用方式：
  from dataset import build_label_schema, build_dataloaders                         # cluener
  from dataset import build_peoples_daily_label_schema, build_peoples_daily_dataloaders  # peoples_daily
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 路径常量
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent.parent
# 数据在原始项目路径下
ORIGINAL_DATA = Path(
    "D:/aipy/AI大模型培训部分/week7序列标注问题_0530/"
    "week7 序列标注问题/序列标注项目/data"
)
DATA_DIR_CLUENER = ORIGINAL_DATA / "cluener"
DATA_DIR_PEOPLES_DAILY = ORIGINAL_DATA / "peoples_daily"


# ══════════════════════════════════════════════════════════════════════════════
# 标签体系构建
# ══════════════════════════════════════════════════════════════════════════════

def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """cluener 数据集 BIO 标签体系：10 类实体 → 21 个标签（O + 2×10）。

    返回 (labels, label2id, id2label)
    """
    entity_types = [
        "address", "book", "company", "game", "government",
        "movie", "name", "organization", "position", "scene",
    ]
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def build_peoples_daily_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """peoples_daily 数据集 BIO 标签体系：3 类实体 → 7 个标签（O + 2×3）。

    实体类型：PER（人名）、ORG（组织机构）、LOC（地名）

    返回 (labels, label2id, id2label)
    """
    entity_types = ["PER", "ORG", "LOC"]
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


# ══════════════════════════════════════════════════════════════════════════════
# BIO ↔ 实体 转换工具
# ══════════════════════════════════════════════════════════════════════════════

def span_to_bio(tokens: list[str], label_dict: dict[str, dict]) -> list[str]:
    """将 cluener span 格式转换为 BIO 标签序列。

    cluener 格式示例：
      {"name": {"张三": [[0,1]]}, "company": {"浙商银行": [[4,7]]}}
    """
    tags = ["O"] * len(tokens)
    for entity_type, surfaces in label_dict.items():
        for surface, positions in surfaces.items():
            for start, end in positions:
                if start < len(tags):
                    tags[start] = f"B-{entity_type}"
                for i in range(start + 1, min(end, len(tags))):
                    tags[i] = f"I-{entity_type}"
    return tags


def bio_tags_to_entities(tokens: list[str], ner_tags: list[str]) -> list[dict]:
    """将 BIO 标签序列反解析为实体列表。

    peoples_daily 格式输入：ner_tags = ["O","B-PER","I-PER","O",...]
    输出：[{"text": "张三", "type": "PER"}, ...]

    此函数供 llm_ner.py、train_sft.py、evaluate_sft.py 共用。
    """
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if current_tokens and current_type:
                entities.append({
                    "text": "".join(current_tokens),
                    "type": current_type,
                })
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            if current_tokens and current_type:
                entities.append({
                    "text": "".join(current_tokens),
                    "type": current_type,
                })
            current_tokens = []
            current_type = None

    if current_tokens and current_type:
        entities.append({
            "text": "".join(current_tokens),
            "type": current_type,
        })

    return entities


# ══════════════════════════════════════════════════════════════════════════════
# cluener Dataset（保持原有实现不变）
# ══════════════════════════════════════════════════════════════════════════════

class CluenerDataset(Dataset):
    """cluener 数据集：从 span 标注转为 BIO 标签序列。

    BERT 子词对齐策略（word_ids）：
      - 中文 BERT 逐字 tokenize，word_ids 从 0 递增（每个字一个 word）
      - 非首子词（word 与上一个相同）→ 标签设 -100
      - [CLS]/[SEP]（word=None）→ 标签设 -100
    """

    def __init__(self, data, tokenizer: BertTokenizer, label2id: dict[str, int],
                 max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["text"]  # cluener 文本是字符串，需 list() 转字符列表
        label_dict = item.get("label", {})

        bio_tags = span_to_bio(list(tokens), label_dict)

        encoding = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        prev_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                labels[i] = self.label2id.get(bio_tags[word_id], 0)
            prev_word_id = word_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
# peoples_daily Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PeoplesDailyDataset(Dataset):
    """peoples_daily 数据集：数据已是 BIO 格式，无需 span_to_bio() 转换。

    数据格式：
      {"tokens": ["海","钓","比","赛",...], "ner_tags": ["O","O","O","O",...]}

    与 CluenerDataset 的关键区别：
      - tokens 已是字符列表，无需 list(text) 转换
      - ner_tags 已是 BIO 标签字符串，直接映射到 label id
      - 不需要 span_to_bio()
    """

    def __init__(self, data, tokenizer: BertTokenizer, label2id: dict[str, int],
                 max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]       # 已是字符列表
        ner_tags = item["ner_tags"]   # 已是 BIO 标签列表

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        # 子词对齐：非首子词设 -100
        labels = torch.full_like(input_ids, -100)
        prev_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                tag = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                labels[i] = self.label2id.get(tag, 0)
            prev_word_id = word_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict[str, int],
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 cluener 的 train / val / test DataLoader。"""
    if data_dir is None:
        data_dir = DATA_DIR_CLUENER

    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    train_ds = CluenerDataset(train_data, tokenizer, label2id, max_length)
    val_ds   = CluenerDataset(val_data,   tokenizer, label2id, max_length)
    test_ds  = CluenerDataset(test_data,  tokenizer, label2id, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_peoples_daily_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict[str, int],
    batch_size: int = 16,
    max_length: int = 128,
    data_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 peoples_daily 的 train / val / test DataLoader。"""
    if data_dir is None:
        data_dir = DATA_DIR_PEOPLES_DAILY

    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    train_ds = PeoplesDailyDataset(train_data, tokenizer, label2id, max_length)
    val_ds   = PeoplesDailyDataset(val_data,   tokenizer, label2id, max_length)
    test_ds  = PeoplesDailyDataset(test_data,  tokenizer, label2id, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
