"""
dataset.py —— 中文命名实体识别（NER）数据处理模块
--------
1. BIO 标签体系构建：从 label_names.json 读取或使用默认列表，构建 label2id / id2label 映射
2. BERT 子词对齐：tokenizer 将一个字符可能拆为多个子词，需要用 word_ids() 对齐标签
   - 首子词：保留原始 BIO 标签（参与 loss 计算）
   - 非首子词 / 特殊 token：标记 -100（PyTorch 的 cross_entropy 会自动忽略）
3. NerDataset：继承 torch.utils.data.Dataset，处理已预分词 + BIO 标注格式的数据
4. build_dataloaders：工厂函数，一键构建 train / val / test 的 DataLoader

使用方式
--------
    from dataset import build_label_schema, build_dataloaders
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    labels, label2id, id2label = build_label_schema()
    train_loader, val_loader, test_loader = build_dataloaders(tokenizer, label2id)
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# ──────────────────────────── 常量 ────────────────────────────

# 模块所在目录的上级目录（即项目根目录）
ROOT = Path(__file__).resolve().parent.parent

# 默认数据目录：项目根目录 / data
DATA_DIR = ROOT / "data" / "peoples_daily"

# 默认标签列表：BIO 标注方案，3 类实体 × 2（B/I）+ O = 7 个标签
DEFAULT_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# ──────────────────────────── BIO 标签体系构建 ────────────────────────────

def build_label_schema(data_dir: Optional[Path] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    构建 BIO 标签体系：标签列表 + 正向映射 + 反向映射。

    优先从 data_dir / label_names.json 读取标签列表；
    若文件不存在，则使用 DEFAULT_LABELS 作为回退。

    Parameters
    ----------
    data_dir : Optional[Path]
        数据目录，默认为 DATA_DIR。函数会尝试读取该目录下的 label_names.json。

    Returns
    -------
    labels : list[str]
        标签名称列表，顺序即为 id 顺序（O→0, B-PER→1, ...）
    label2id : dict[str, int]
        标签字符串 → 整数 id 的映射
    id2label : dict[int, str]
        整数 id → 标签字符串的映射
    """
    # 确定数据目录
    if data_dir is None:
        data_dir = DATA_DIR

    # 尝试从 label_names.json 读取标签列表
    label_file = data_dir / "label_names.json"
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"[build_label_schema] 从 {label_file} 读取标签列表")
    else:
        # 文件不存在时使用默认列表，保证鲁棒性
        labels = DEFAULT_LABELS
        print(f"[build_label_schema] label_names.json 不存在，使用默认标签列表")

    # 构建双向映射
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}

    # 打印标签体系，方便调试和验证
    print("=" * 50)
    print("标签体系 (BIO Schema)")
    print("=" * 50)
    for label, idx in label2id.items():
        print(f"  {label:<8s} → {idx}")
    print(f"  标签总数: {len(labels)}")
    print("=" * 50)

    return labels, label2id, id2label


# ──────────────────────────── 辅助函数：加载记录 ────────────────────────────

def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """
    从 JSON 文件加载指定 split 的数据记录。

    Parameters
    ----------
    split : str
        数据集划分，取值 "train" / "validation" / "test"
    data_dir : Optional[Path]
        数据目录，默认为项目根目录 / data / peoples_daily

    Returns
    -------
    records : list
        JSON 记录列表，每条记录包含 tokens 和 ner_tags
    """
    if data_dir is None:
        data_dir = DATA_DIR

    file_path = data_dir / f"{split}.json"
    print(f"[load_records] 加载 {file_path} ...")

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[load_records] {split:12s} → {len(records):>6d} 条记录")
    return records


# ──────────────────────────── NER Dataset 类 ────────────────────────────

class NerDataset(Dataset):
    """
    中文命名实体识别数据集。

    与 cluener2020 格式的关键区别：
    不需要 span→BIO 转换，tokens 和 ner_tags 已经一一对应。
    但仍需要 BERT 子词对齐——这是本类的核心处理逻辑。

    Parameters
    ----------
    records : list
        JSON 记录列表，每条记录包含 "tokens" 和 "ner_tags"
    tokenizer : BertTokenizer
        BERT 分词器（如 bert-base-chinese）
    label2id : dict
        标签到 id 的映射，用于将字符串标签转为整数
    max_length : int
        最大序列长度，超过则截断，不足则填充
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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        获取单条样本，经过子词对齐后的模型输入。

        核心流程：
        1. 取出 tokens 和 ner_tags
        2. 将字符串标签转为 id 列表
        3. 调用 tokenizer（is_split_into_words=True）
        4. 子词对齐：首子词保留标签，非首子词/特殊token标记 -100
        5. 返回 input_ids, attention_mask, token_type_ids, labels
        """
        # ── 1. 从 record 取出 tokens 和 ner_tags ──
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row.get("ner_tags", [])  # 防御性编码：空值保护

        # ── 2. 将字符串标签转为 id 列表 ──
        # 使用 .get(t, 0) 做防御性编码：未知标签回退到 O（id=0）
        tag_ids = [self.label2id.get(t, 0) for t in ner_tags]

        # ── 3. 调用 tokenizer ──
        # is_split_into_words=True：告诉 tokenizer 输入已经是预分词结果
        # 这样 word_ids() 返回的索引就能与 tokens 列表的索引对齐
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # ── 4. 子词对齐——最关键的技术点 ──
        # word_ids() 返回每个子词对应的原始 token 索引
        # 特殊 token ([CLS]/[SEP]/[PAD]) 对应 None
        # 若某字符被拆为多子词，则这些子词共享同一个 word_id
        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        prev_word_id = None  # 追踪上一个 word_id，用于区分首子词和非首子词

        for wid in word_ids:
            if wid is None:
                # 特殊 token（[CLS]/[SEP]/[PAD]）：不参与 loss 计算
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首子词：该 word_id 首次出现，取对应的 BIO 标签 id
                if wid < len(tag_ids):
                    aligned_labels.append(tag_ids[wid])
                else:
                    # 越界保护：truncation 可能导致 word_id 超出 tag_ids 范围
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 非首子词（同一字符的续接子词）：标记为 -100
                # 为什么标 -100：PyTorch 的 F.cross_entropy(ignore_index=-100)
                # 会自动跳过标签为 -100 的位置，不纳入 loss 计算
                # CRF 的 mask 也只关注 attention_mask=1 的位置
                aligned_labels.append(-100)

        # ── 5. 返回字典 ──
        # squeeze(0) 的必要性：return_tensors="pt" 让 tokenizer 返回 (1, seq_len) 的张量
        # 但 Dataset 的 __getitem__ 应返回单个样本，所以需要去掉 batch 维度 → (seq_len,)
        # DataLoader 会自动在 batch 维度堆叠
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ──────────────────────────── DataLoader 工厂函数 ────────────────────────────

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / validation / test 三个 DataLoader。

    Parameters
    ----------
    tokenizer : BertTokenizer
        BERT 分词器
    label2id : dict
        标签到 id 的映射
    batch_size : int
        批大小，默认 32
    max_length : int
        最大序列长度，默认 128
    data_dir : Optional[Path]
        数据目录，默认为项目根目录 / data / peoples_daily

    Returns
    -------
    train_loader, val_loader, test_loader : tuple[DataLoader, DataLoader, DataLoader]
        三个 DataLoader，训练集 shuffle=True，验证/测试集 shuffle=False
    """
    # 加载三个 split 的数据
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    # 构建三个 Dataset 实例
    train_dataset = NerDataset(train_records, tokenizer, label2id, max_length)
    val_dataset = NerDataset(val_records, tokenizer, label2id, max_length)
    test_dataset = NerDataset(test_records, tokenizer, label2id, max_length)

    # 构建三个 DataLoader
    # 训练集需要打乱顺序以获得更好的训练效果
    # 验证/测试集不打乱，保证结果可复现
    # num_workers=0：Windows 兼容，避免多进程 DataLoader 的问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 打印数据集规模
    print("=" * 50)
    print("数据集规模")
    print("=" * 50)
    print(f"  训练集: {len(train_dataset):>6d} 条, {len(train_loader):>4d} 批")
    print(f"  验证集: {len(val_dataset):>6d} 条, {len(val_loader):>4d} 批")
    print(f"  测试集: {len(test_dataset):>6d} 条, {len(test_loader):>4d} 批")
    print(f"  批大小: {batch_size}")
    print(f"  最大序列长度: {max_length}")
    print("=" * 50)

    return train_loader, val_loader, test_loader
