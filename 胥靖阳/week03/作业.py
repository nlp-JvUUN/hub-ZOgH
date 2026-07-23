# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/5/7 17:53  
@脚本说明 : 

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import Counter

# -------------------- 1. 生成数据集 --------------------
# 为了保证“你”只出现一次，我们首先生成其他四个位置的字符，再在随机位置插入“你”
# 定义常用汉字池（不含“你”），用于填充非“你”位置
CHAR_POOL = [chr(i) for i in range(0x4e00, 0x4e00 + 100)]  # 前100个常用汉字
CHAR_POOL = [c for c in CHAR_POOL if c != '你']  # 移除“你”
# 若不够，补充一些常见字
CHAR_POOL += ['我', '他', '她', '好', '的', '了', '是', '不', '人', '有']
CHAR_POOL = list(set(CHAR_POOL))  # 去重


def generate_sample():
    """生成一个五字文本及其标签（'你'所在索引0~4）"""
    # 随机决定'你'的位置
    pos = random.randint(0, 4)
    chars = []
    for i in range(5):
        if i == pos:
            chars.append('你')
        else:
            chars.append(random.choice(CHAR_POOL))
    text = ''.join(chars)
    label = pos
    return text, label


def generate_dataset(num_samples):
    texts = []
    labels = []
    for _ in range(num_samples):
        t, l = generate_sample()
        texts.append(t)
        labels.append(l)
    return texts, labels


# 生成训练集和测试集
train_texts, train_labels = generate_dataset(5000)
test_texts, test_labels = generate_dataset(1000)

# 展示部分训练样本
print("训练样本示例（文本 -> 标签（位置））：")
for i in range(10):
    print(f"  {train_texts[i]} -> {train_labels[i]}")


# -------------------- 2. 构建词汇表 --------------------
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(list(text))
    # 按频次排序，预留PAD（0）和UNK（1）
    vocab = {char: idx + 2 for idx, (char, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


vocab = build_vocab(train_texts)
idx2char = {idx: char for char, idx in vocab.items()}
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")


# 文本转索引序列（固定长度5）
def text_to_indices(text, vocab, max_len=5):
    indices = [vocab.get(ch, vocab['<UNK>']) for ch in text]
    # 补全或截断（这里本就5个字，无需操作）
    return indices[:max_len] + [vocab['<PAD>']] * (max_len - len(indices))


# -------------------- 3. 创建PyTorch Dataset --------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_indices(self.texts[idx], self.vocab)
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


batch_size = 64
train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# -------------------- 4. 定义模型（LSTM，可切换为RNN） --------------------
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, use_lstm=True):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if use_lstm:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh', dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        out, _ = self.rnn(embedded)  # out: (batch, seq_len, hidden_dim)
        # 取最后一个时间步的输出
        last_out = out[:, -1, :]  # (batch, hidden_dim)
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits


# 超参数
embed_dim = 64
hidden_dim = 128
num_layers = 2
num_classes = 5
use_lstm = True  # 改为False则使用RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes, num_layers, use_lstm=use_lstm).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"模型类型: {'LSTM' if use_lstm else 'RNN'}")
print(model)


# -------------------- 5. 训练函数 --------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# -------------------- 6. 训练循环 --------------------
epochs = 20
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


# -------------------- 7. 简单推理示例 --------------------
def predict(text, model, vocab, device):
    model.eval()
    indices = text_to_indices(text, vocab)
    tensor = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return pred


print("\n推理示例：")
test_samples = ["你我他她它", "他你好吗", "你好世界", "这是你好", "好你你好"]  # 注意第5个有俩“你”
for s in test_samples:
    pred = predict(s, model, vocab, device)
    # (实际位置取决于首个'你'？训练时只含单个'你')
    print(f"'{s}' -> 预测位置: {pred}")