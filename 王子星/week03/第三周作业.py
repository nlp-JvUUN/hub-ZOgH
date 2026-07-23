import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 超参数
SEED = 42
N_SAMPLES = 5000
MAXLEN = 5  # 文本固定 5 个字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 25
TRAIN_RATIO = 0.8
N_CLASSES = 5  # 位置 0~4
MODEL_PATH = 'test_RNN_model.bin'
VOCAB_PATH = 'test_RNN_vocab.json'

random.seed(SEED)
torch.manual_seed(SEED)

# 数据生成
# 填充非"你"位置的常用汉字（不含"你"）
OTHER_CHARS = '这是一很好天大人个中上下左右前后内外东西南北新旧高低大小多少美丑真假黑白红绿黄蓝紫青灰金木水火土风花雪月山河湖海云雨电气'


def make_sample():
    """生成一个 5 字文本，"你"出现在随机位置（0~4）"""
    pos = random.randint(0, 4)
    chars = [random.choice(OTHER_CHARS) for _ in range(5)]
    chars[pos] = '你'
    return ''.join(chars), pos


def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
    random.shuffle(data)
    return data


# 词表构建与编码
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# 格式化文本数据
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]     # 类别索引 0~4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),      # CrossEntropy 需要 long 型标签
        )


# 模型定义
class PositionRNN(nn.Module):
    """
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear(5)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, N_CLASSES)   # 5 类输出

    def forward(self, x):
        # x: (batch, 5)
        e, _ = self.rnn(self.embedding(x))  # (B, 5, hidden_dim)
        pooled = e.max(dim=1)[0]    # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)   # (B, 5)   logits，无 sigmoid
        return out


# 训练
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)   # (B, 5)
            pred = logits.argmax(dim=1)     # (B,)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def test_model(model, vocab):
    """使用一组测试文本进行推理演示"""
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你喜欢吗',
        '爱你好吗',
        '你好世界',
        '大家爱你',
        '祝福你我',
        '你你我我',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)[0]
            probs  = torch.softmax(logits, dim=0)
            pred   = logits.argmax().item()
            print(f"  预测位置：{pred}（'你'在 index=？） 文本：{sent}  |  各类概率：{[f'{p:.2f}' for p in probs.tolist()]}")


def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()               # 多分类交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)   # (B, 5)
            loss = criterion(logits, y)     # y 是类别索引 (B,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # 保存最新模型
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    print(f"最新模型已保存至：{MODEL_PATH}")

    # 保存词表
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词表已保存至：{VOCAB_PATH}")

    test_model(model, vocab)


if __name__ == '__main__':
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        model = PositionRNN(vocab_size=len(vocab))
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"加载已有模型（词表大小：{len(vocab)}），直接推理\n")
        test_model(model, vocab)
    else:
        train()


