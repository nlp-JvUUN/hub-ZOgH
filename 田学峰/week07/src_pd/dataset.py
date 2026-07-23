"""
人民日报 NER 数据集：token-level BIO 标注 + BERT 子词对齐

与 cluener 数据格式的区别：
  - cluener：span 格式 {"name": {"叶老桂": [[9, 11]]}}，需 span→BIO 转换
  - peoples_daily：已是 token-level BIO，直接用 ner_tags 字段

数据格式示例：
  {"tokens": ["海", "钓", ...], "ner_tags": ["O", "O", ..., "B-LOC", "I-LOC", ...]}

实体类型：PER（人名）、ORG（组织机构）、LOC（地名）

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
OUTPUT_DIR = ROOT / "outputs_pd"

# peoples_daily 标签体系（3 类实体 × 2 + O = 7）
ENTITY_TYPES = ["PER", "ORG", "LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 peoples_daily BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据已是 token-level BIO 标注，直接将 ner_tags 转为 id，
    再用 BertTokenizer 的 word_ids() 做子词对齐。
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        # 1. 将字符串标签转为 id
        char_labels = [self.label2id.get(t, 0) for t in ner_tags]

        # 2. tokens 已是字符列表，直接传入 tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：非首子词和特殊 token 标记为 -100
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载 peoples_daily 数据集的 JSON 文件。"""
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
