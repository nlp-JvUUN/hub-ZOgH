"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. peoples_daily 的 BIO token-tag 格式直接编码
     - 记录: {"tokens": [...], "ner_tags": ["O", "B-PER", ...]}
  3. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  4. DataLoader 工厂函数统一封装，通过 --dataset 切换

使用方式：
  from dataset import build_label_schema, build_dataloaders
  labels, label2id, id2label = build_label_schema()  # 默认 CLUENER
  train_loader, val_loader, test_loader = build_dataloaders(
      tokenizer, label2id, dataset_name="peoples_daily"
  )
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from dataset_config import get_config, build_label_schema as _build_label_schema

ROOT = Path(__file__).parent.parent


def build_label_schema(entity_types: Optional[list[str]] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。

    不传 entity_types 时默认使用 CLUENER 的 10 类实体（向后兼容）。
    """
    if entity_types is None:
        entity_types = get_config("cluener")["entity_types"]
    return _build_label_schema(entity_types)


def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    """将 cluener2020 的 span 格式标注转为逐字符 BIO 标签 id 列表。

    教学要点：先全部初始化为 O，再按 span 位置填入 B/I。
    若存在嵌套实体（本数据集极少），外层实体覆盖内层。
    """
    n = len(text)
    bio = ["O"] * n

    if not label_dict:
        return [label2id[t] for t in bio]

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                if start >= n or end >= n:
                    continue
                bio[start] = b_tag
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag

    return [label2id.get(t, 0) for t in bio]


def _subword_align_labels(
    word_ids: list,
    char_labels: list[int],
    label2id: dict,
) -> list[int]:
    """BERT 子词对齐：非首子词和特殊 token 标记为 -100。

    与 CluenerDataset 和 PeoplesDailyDataset 共用。
    """
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
    return aligned_labels


class CluenerDataset(Dataset):
    """cluener2020 的 PyTorch Dataset。

    教学流程：
      text → span_to_bio → 字符级 BIO ids
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
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
        text: str = row["text"]
        label_dict: dict = row.get("label") or {}

        # 1. span → 字符级 BIO id 列表
        char_labels = span_to_bio(text, label_dict, self.label2id)

        # 2. 将文本拆为字符列表，传入 tokenizer
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = _subword_align_labels(word_ids, char_labels, self.label2id)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    peoples_daily 格式：{"tokens": [...], "ner_tags": ["O", "B-PER", ...]}

    教学流程：
      ner_tags → label ids（直接映射，无需 span→BIO 转换）
              → BertTokenizer (is_split_into_words=True)
              → 用 word_ids() 对齐子词标签（非首子词设为 -100）
              → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 256,
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

        # 1. BIO 标签字符串 → label id 列表（一字一标签）
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 将字符列表传入 tokenizer（中文一字一 token）
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = _subword_align_labels(word_ids, char_labels, self.label2id)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载数据集 split 的 JSON 记录。"""
    d = data_dir or Path(get_config("cluener")["data_dir"])
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
    dataset_name: str = "cluener",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。

    Args:
        tokenizer: BERT tokenizer
        label2id: 标签名 → id 映射
        batch_size: 批大小
        max_length: 最大序列长度
        data_dir: 数据目录（可选，优先于 dataset_name）
        dataset_name: 数据集名称，"cluener" 或 "peoples_daily"
    """
    cfg = get_config(dataset_name)
    d = data_dir or Path(cfg["data_dir"])
    fmt = cfg["format"]
    default_ml = cfg.get("max_length_default", 128)
    if max_length == 128 and default_ml != 128:
        # 如果用户未指定 max_length（用了默认值 128），使用数据集的推荐值
        # 这只在调用方传了显式的 max_length 时才用调用方的值
        pass

    train_records = load_records("train", d)
    val_records = load_records("validation", d)
    test_records = load_records("test", d)

    DatasetClass = PeoplesDailyDataset if fmt == "bio" else CluenerDataset
    # 使用数据集的推荐 max_length（除非调用方显式覆盖）
    effective_ml = max_length

    train_ds = DatasetClass(train_records, tokenizer, label2id, effective_ml)
    val_ds = DatasetClass(val_records, tokenizer, label2id, effective_ml)
    test_ds = DatasetClass(test_records, tokenizer, label2id, effective_ml)

    print(f"数据集 [{dataset_name}]：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}"
          f"（格式={fmt}，max_length={effective_ml}）")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
