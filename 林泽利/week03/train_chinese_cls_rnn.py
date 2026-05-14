import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本

设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
可以选择如下任务: 对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss (交叉熵函数)

"""

# ─── 超参数 ────────────────────────────────────────────────
vocab = list("测试你我他它上下左右中前后的得了吗呢吧哦哈吗呀今天下雨出门忘带雨伞")

SEED = 42
N_SAMPLES = 4000
MAXLEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 1. 生成5字样本：含一个“你” =====================
def build_sample():
    """随机生成一个文本包含5个汉字，获取一个随机值存放‘你’字，返回：文本, 标签(你所在位置0~4)"""
    # 随机选你出现的位置
    pos = random.randint(0, 4)
    # 生成生成一个文本包含5个汉字
    chars = random.choices(vocab, k=5)
    # 随机选的位置固定为"你"
    chars[pos] = "你"
    text = "".join(chars)
    return text, pos

def build_dataset(n_samples):
    # 随机生成样本
    data = []
    for _ in range(n_samples):
        text, pos = build_sample()
        data.append((text, pos))
    random.shuffle(data)
    return data

# ===================== 2. 构建词表 & 编码 =====================
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

# ===================== 3. Dataset / DataLoader  =====================
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)  # 标签必须是long
        )

# ===================== 4. 模型定义 =====================
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 输出5类！

    def forward(self, x):
        e = self.embedding(x)    # (B, seq, embed)
        out, h = self.rnn(e)     # out: (B, seq, hidden)
        h = h.squeeze(0)         # (B, hidden)
        logits = self.fc(h)      # (B, 5) → 5分类
        return logits

# ===================== 5. 评估（多分类准确率） =====================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

# ===================== 6. 训练 =====================
def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # ===================== 推理测试 =====================
    print("\n===== 推理测试（你在第几位） =====")
    model.eval()
    test_sents = [
        "你上下左右",
        "中你吧哦哈",
        "今你下雨出",
        "得了吗你雨",
        "今天下雨你"
    ]

    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_pos = torch.argmax(logits).item()
            real_pos = sent.index("你")
            print(f"文本：{sent} | 真实位置：{real_pos} | 预测位置：{pred_pos}")

if __name__ == '__main__':
    train()
