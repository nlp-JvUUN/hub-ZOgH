# 洪家明week2作业

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

'''

【第三周作业内容：】
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

'''

# 设置随机种子
torch.manual_seed(42)
random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ========== 1. 生成数据 ==========
def generate_data(num_samples=5000):
    """生成5字文本，'你'在第几位就是第几类（0-4）"""
    chars = '你我他风火水地光暗'
    texts, labels = [], []

    for _ in range(num_samples):
        pos = random.randint(0, 4)  # '你'的位置 0-4
        text = []
        for i in range(5):
            if i == pos:
                text.append('你')
            else:
                text.append(random.choice(chars.replace('你', '')))
        texts.append(''.join(text))
        labels.append(pos)
    return texts, labels


# 生成数据
train_texts, train_labels = generate_data(4000)
val_texts, val_labels = generate_data(500)
test_texts, test_labels = generate_data(500)

# 构建词表
vocab = {'<PAD>': 0, '<UNK>': 1}
for text in train_texts:
    for char in text:
        if char not in vocab:
            vocab[char] = len(vocab)
print(f"词表大小: {len(vocab)}")
print(f"词表内容: {vocab}")


# 文本转索引
def encode(text, max_len=5):
    indices = [vocab.get(c, 1) for c in text]
    return indices + [0] * (max_len - len(indices))


# 准备数据
train_x = torch.LongTensor([encode(t) for t in train_texts])
train_y = torch.LongTensor(train_labels)
val_x = torch.LongTensor([encode(t) for t in val_texts])
val_y = torch.LongTensor(val_labels)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

print(f"\n示例: {train_texts[0]} -> 类别 {train_labels[0] + 1}")


# ========== 2. 定义模型 ==========
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq, embed)
        _, h = self.rnn(x)  # h: (1, batch, hidden)
        return self.fc(h.squeeze(0))  # (batch, 5)


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


# ========== 3. 训练函数 ==========
def train(model, train_loader, val_loader, epochs=30, lr=0.05):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 训练
        model.train()
        train_correct, train_total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, pred = output.max(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:2d}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model


# ========== 4. 测试函数 ==========
def test_model(model, test_texts, test_labels):
    model.eval()
    correct = 0
    for text, label in zip(test_texts, test_labels):
        x = torch.LongTensor([encode(text)]).to(device)
        with torch.no_grad():
            output = model(x)
            pred = output.argmax().item()
        if pred == label:
            correct += 1
    return correct / len(test_texts)


# ========== 5. 训练并测试 ==========
print("\n" + "=" * 50)
print("训练 RNN 模型")
print("=" * 50)
rnn_model = train(SimpleRNN(len(vocab)), train_loader, val_loader)

print("\n" + "=" * 50)
print("训练 LSTM 模型")
print("=" * 50)
lstm_model = train(SimpleLSTM(len(vocab)), train_loader, val_loader)

# 在测试集上评估
rnn_acc = test_model(rnn_model, test_texts, test_labels)
lstm_acc = test_model(lstm_model, test_texts, test_labels)

print("\n" + "=" * 50)
print("测试结果")
print("=" * 50)
print(f"RNN 准确率:  {rnn_acc:.4f} ({rnn_acc * 100:.2f}%)")
print(f"LSTM 准确率: {lstm_acc:.4f} ({lstm_acc * 100:.2f}%)")

# 演示预测
print("\n" + "=" * 50)
print("预测示例 (LSTM模型)")
print("=" * 50)
test_samples = ['你好世界光', '你好你你好', '水火你好地', '你我他她是']
for text in test_samples:
    x = torch.LongTensor([encode(text)]).to(device)
    with torch.no_grad():
        pred = lstm_model(x).argmax().item()
    print(f"'{text}' -> '你'在第 {pred + 1} 位")