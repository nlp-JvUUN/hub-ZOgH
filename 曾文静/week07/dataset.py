# -*- coding: utf-8 -*-
"""
week07/dataset.py
=================
cluener2020 数据集的加载与标签对齐

核心流程:
  1. span 标注 -> 字级 BIO 标签   (span_to_bio)
  2. 字符 -> BERT 子词, 标签对齐  (NERDataset)
     - 特殊 token([CLS]/[SEP]/[PAD]): Linear 用 -100 屏蔽, CRF 用 0(O) 填充
     - 非首子词: Linear 用 -100 屏蔽, CRF 沿用首子词标签
"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# cluener2020 的 10 类细粒度实体
ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
]


def build_label_schema():
    """BIO 标签体系: O + 每类实体{B-, I-} = 21 个标签"""
    labels = ["O"]
    for t in ENTITY_TYPES:
        labels += [f"B-{t}", f"I-{t}"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def span_to_bio(text: str, label_dict: dict):
    """
    把 cluener 的 span 标注转成字级 BIO 标签。
    cluener 的 span 是闭区间 [start, end](两端都包含):
      {"text": "张三在北京", "label": {"name": {"张三": [[0, 1]]},
                                        "address": {"北京": [[2, 3]]}}}
    返回 (tokens, ner_tags)
    """
    tokens = list(text)
    ner_tags = ["O"] * len(tokens)

    for ent_type, spans_by_text in (label_dict or {}).items():
        for ent_text, spans in spans_by_text.items():
            for s, e in spans:
                if s < 0 or e >= len(tokens) or s > e:
                    continue
                # 边界与实体文本不一致时, 尝试按文本自动修正位置
                if text[s:e + 1] != ent_text:
                    idx = text.find(ent_text)
                    if idx >= 0:
                        s, e = idx, idx + len(ent_text) - 1
                    else:
                        print(f"[警告] 实体文本不匹配: '{text[s:e+1]}' != '{ent_text}', 已跳过")
                        continue
                ner_tags[s] = f"B-{ent_type}"
                for i in range(s + 1, e + 1):
                    ner_tags[i] = f"I-{ent_type}"
    return tokens, ner_tags


class NERDataset(Dataset):
    """PyTorch 数据集: 返回 BERT 输入 + 两套标签"""

    def __init__(self, records, tokenizer, label2id, max_length=128):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        # 预转换: span -> BIO, 同时保留原始信息供评估用
        self.tokens_list, self.true_tags, self.word_ids = [], [], []
        self.samples = []
        for rec in records:
            tokens, tags = span_to_bio(rec["text"], rec.get("label"))
            self.tokens_list.append(tokens)
            self.true_tags.append(tags)
            self.samples.append((tokens, tags))
        # 先做一次编码, 缓存 word_ids(评估对齐也需要)
        for tokens, _ in self.samples:
            enc = self.tokenizer(tokens, is_split_into_words=True,
                                 truncation=True, max_length=max_length)
            self.word_ids.append(enc.word_ids())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, tags = self.samples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True,
                             truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        word_ids = enc.word_ids()

        linear_labels, crf_labels = [], []
        prev_wid = None
        for wid in word_ids:
            if wid is None:                      # 特殊 token / padding
                linear_labels.append(-100)
                crf_labels.append(0)             # CRF 需要合法标签, 填 O
            elif wid != prev_wid:                # 首子词: 取真实标签
                lid = self.label2id.get(tags[wid], 0)
                linear_labels.append(lid)
                crf_labels.append(lid)
            else:                                # 非首子词
                linear_labels.append(-100)       # Linear 忽略
                crf_labels.append(crf_labels[-1])  # CRF 沿用首子词标签
            prev_wid = wid

        return (enc["input_ids"][0], enc["attention_mask"][0],
                enc["token_type_ids"][0],
                torch.tensor(linear_labels, dtype=torch.long),
                torch.tensor(crf_labels, dtype=torch.long))


def load_records(data_dir, split):
    """读取 {split}.json -> records 列表"""
    path = Path(data_dir) / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(data_dir, tokenizer, label2id, batch_size=16,
                      max_length=128, num_train=None, num_workers=0):
    """
    构建 train/validation/test 三个 DataLoader。
    num_train: 限制训练样本数(快速实验用), None 表示全部
    """
    train_records = load_records(data_dir, "train")
    if num_train is not None:
        train_records = train_records[:num_train]
        print(f"[提示] 仅使用前 {num_train} 条训练样本(快速实验)")

    datasets = {
        "train": NERDataset(train_records, tokenizer, label2id, max_length),
        "validation": NERDataset(load_records(data_dir, "validation"),
                                 tokenizer, label2id, max_length),
        "test": NERDataset(load_records(data_dir, "test"),
                           tokenizer, label2id, max_length),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers),
    }
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} 条")
    return loaders
