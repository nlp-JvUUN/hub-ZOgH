import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

"""
文本五分类

任务说明：
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
“你”出现在第 1 个位置，标签就是第 1 类；
“你”出现在第 2 个位置，标签就是第 2 类；
依此类推，一共 5 个类别。
"""

random.seed(42)
torch.manual_seed(42)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 随机词表：包含常用汉字，用于生成训练语料
CHARS = [
    '我', '你', '他', '她', '它', '们', '的', '了', '在', '是',
    '有', '不', '这', '个', '人', '上', '来', '到', '时', '大',
    '去', '说', '好', '看', '想', '做', '要', '会', '能', '可',
    '多', '少', '年', '月', '日', '天', '水', '火', '土', '木',
    '金', '山', '河', '海', '风', '雨', '雪', '花', '草', '树',
    '鸟', '鱼', '虫', '马', '牛', '羊', '鸡', '狗', '猫', '鼠',
    '红', '黄', '蓝', '绿', '白', '黑', '大', '小', '多', '少',
    '快', '慢', '新', '旧', '高', '低', '长', '短', '宽', '窄',
    '美', '丑', '好', '坏', '真', '假', '对', '错', '是', '非',
    '东', '南', '西', '北', '中', '左', '右', '前', '后', '上',
]

def build_vocab(data):
    """
    构建词表

    Args:
        data: 训练数据集

    Returns:
        dict: 词表，包含字符到索引的映射
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=5):
    """
    将文本编码为索引序列

    Args:
        sent: 输入文本
        vocab: 词表
        maxlen: 最大序列长度

    Returns:
        list: 索引序列
    """
    ids  = [vocab.get(ch, 1) for ch in sent]  # 1是<UNK>的索引
    ids  = ids[:maxlen]  # 截断到最大长度
    ids += [0] * (maxlen - len(ids))  # 填充到最大长度
    return ids

def generate_sample():
    """
    生成一个包含"你"字的五字文本，并返回其标签（"你"字的位置）

    Returns:
        tuple: (文本, 标签)，标签为0-4的整数，表示"你"字在文本中的位置（从0开始）
    """
    # 随机选择"你"字的位置（1-5）
    pos = random.randint(1, 5)  # 位置从1开始计数

    # 生成四个其他字符
    other_chars = [random.choice(CHARS) for _ in range(4)]

    # 构建5字文本
    text_list = list(other_chars)
    text_list.insert(pos - 1, '你')  # 在指定位置插入"你"字
    text = ''.join(text_list)

    return text, pos - 1  # 返回文本和标签（标签从0开始，即0-4）

def build_dataset(n=5000):
    """
    构建训练数据集

    Args:
        n: 样本数量

    Returns:
        list: 包含(文本, 标签)元组的列表
    """
    data = []
    for _ in range(n):
        text, label = generate_sample()
        data.append((text, label))
    random.shuffle(data)
    return data

class TextDataset(Dataset):
    """
    文本数据集类
    """
    def __init__(self, data, vocab):
        """
        初始化数据集

        Args:
            data: 训练数据集
            vocab: 词表
        """
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 标签使用long类型，用于交叉熵损失
        )
    
class MultiClassficationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64, class_num=5):
        super(MultiClassficationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size * 2, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        pooled = x.max(dim=1)[0]
        pooled = self.layer_norm(pooled)
        x = self.dropout(pooled)
        out = self.linear(x)
 
        return out

def evaluate(model, loader):
    """
    评估模型性能

    Args:
        model: 模型
        loader: 数据加载器

    Returns:
        float: 准确率
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)  # 获取模型输出
            pred = logits.argmax(dim=1)  # 获取预测类别
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train():
    """
    训练函数
    """
    print("生成数据集...")
    data  = build_dataset(3000)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 划分训练集和验证集
    split      = int(len(data) * 0.8)
    train_data = data[:split]
    val_data   = data[split:]

    # 创建数据加载器
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=64)

    # 初始化模型
    model = MultiClassficationModel(vocab_size=len(vocab))

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 训练循环
    for epoch in range(1, 20 + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            # 前向传播
            logits = model(X)

            # 计算损失
            loss = criterion(logits, y)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{20}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 输出最终验证准确率
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你我他它们',  # "你"在第1位
        '我你他它们',  # "你"在第2位
        '我他你它们',  # "你"在第3位
        '我他它你们',  # "你"在第4位
        '我他它们你',  # "你"在第5位
        '你是谁',
        '谁是你的',
        '我他它们是你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = nn.functional.softmax(logits, dim=1)  # 获取各类别的概率
            pred_class = logits.argmax(dim=1).item() + 1  # 预测类别（1-5）
            prob_max = prob[0][pred_class-1].item()  # 预测类别的概率
            print(f"  [预测位置: {pred_class}, 概率: {prob_max:.4f}]  {sent}")


if __name__ == '__main__':
    train()


