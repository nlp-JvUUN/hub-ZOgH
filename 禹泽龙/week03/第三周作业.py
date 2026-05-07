"""
rnnWorkDemo.py
任务：判断句子中 '你' 字出现在第几位（共10类：第1-10位）
模型：Embedding → RNN → MaxPool → BN → Dropout → Linear → CrossEntropyLoss
优化：Adam (lr=1e-3)   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
NUM_CLASSES = 10          # '你'在第几位(0-9)，共10类
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 数据生成 ───────────────────────────────────────────────
CHARS = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多信思群还'
                                                        # 排除'你'字

def generate_sample():
    pos = random.randint(0, 9)        # '你'的位置（0-9）
    length = MAXLEN

    # 生成不含'你'的9个字
    chars = [random.choice(CHARS) for _ in range(length - 1)]
    chars.insert(pos, '你')            # 在指定位置插入'你'

    sentence = ''.join(chars)
    return sentence, pos               # 句子, 类别标签(0-9)


def build_dataset(n=N_SAMPLES):
    return [generate_sample() for _ in range(n)]


# ─── 词表 & 编码 ────────────────────────────────────────────
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


# ─── Dataset ────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 模型 ──────────────────────────────────────────────────
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)
        return out


# ─── 训练 ──────────────────────────────────────────────────
def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model     = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        val_acc = correct / total

        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{val_acc:.4f}")

    # ─── 推理示例 ────────────────────────────────────────────
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = ['你在这的第几位', '我在第你位测试', '不你一样的测试']
    with torch.no_grad():
        for sent in test_sents:
            ids  = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            pred = model(ids).argmax(dim=1).item() + 1
            print(f"  句子:「{sent}」  →  '你'在第 {pred} 位")


if __name__ == '__main__':
    train()
