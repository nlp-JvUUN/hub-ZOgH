import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
train_chinese_cls_rnn_lstm.py
改造任务：5 字文本中“你”在第几位 → 5分类任务
模型：RNN / LSTM 自由切换
分类：1/2/3/4/5 类
"""

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 8000  # 数据量
SEQ_LEN = 5  # 固定5个字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_RATIO = 0.8
USE_LSTM = True  # True=LSTM，False=RNN

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 生成 5 字数据：必须含“你”，位置对应标签 0~4（对应1~5位）────────────
CHARS = [
    '我', '他', '她', '它', '们', '这', '那', '来', '去', '到',
    '吃', '喝', '看', '听', '说', '走', '跑', '爱', '恨', '想'
]


def generate_5_sentence():
    pos = random.randint(0, 4)  # 0~4 对应第1~5位
    sent = [''] * 5
    sent[pos] = '你'
    for i in range(5):
        if sent[i] == '':
            sent[i] = random.choice(CHARS)
    return ''.join(sent), pos  # 返回句子 + 标签(0-4)


def build_dataset(n):
    data = []
    for _ in range(n):
        data.append(generate_5_sentence())
    return data


# ─── 2. 构建词表 & 编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for c in sent:
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab, seq_len=SEQ_LEN):
    ids = [vocab.get(c, 1) for c in sentence]
    ids = ids[:seq_len] + [0] * (seq_len - len(ids))
    return ids


# ─── 3. Dataset ────────────────────────────────────────────
class FiveTextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode_sentence(s, vocab) for s, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])


# ─── 4. 模型：RNN / LSTM 5分类 ───────────────────────────────
class TextRNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5, use_lstm=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if use_lstm:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 5)
        x = self.embedding(x)  # (B,5,D)

        # RNN/LSTM 前向
        out, _ = self.rnn(x)  # (B,5,H)

        # 取最后时刻输出做分类
        last = out[:, -1, :]  # (B,H)

        logits = self.fc(last)  # (B,5)
        return logits


# ─── 5. 训练 & 评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train():
    print("正在生成 5 字数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    vocab_size = len(vocab)
    print(f"样本数：{len(data)} | 词表大小：{vocab_size}")

    # 划分训练/验证
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    # 加载器
    train_loader = DataLoader(FiveTextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FiveTextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 模型
    model = TextRNNLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=5,
        use_lstm=USE_LSTM
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model_name = "LSTM" if USE_LSTM else "RNN"
    print(f"模型：{model_name} | 5分类任务\n")

    # 开始训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | val_acc={acc:.4f}")

    print("\n训练完成！最终准确率：", round(evaluate(model, val_loader), 4))

    # ─── 测试推理 ─────────────────────────────────────
    print("\n=== 测试示例（5字句子，预测“你”在第几位） ===")
    test_sents = [
        "我你他她它",  # 第2位 → 标签1
        "他她你我它",  # 第3位 → 标签2
        "我爱你中国",  # 第3位 → 标签2
        "你真好呀哦",  # 第1位 → 标签0
        "今天你开心",  # 第3位 → 标签2
        "真的好想你"  # 第5位 → 标签4
    ]

    model.eval()
    with torch.no_grad():
        for s in test_sents:
            ids = encode_sentence(s, vocab)
            x = torch.tensor([ids])
            logits = model(x)
            pred = torch.argmax(logits).item()
            pos = pred + 1
            print(f"句子：{s} → 预测“你”在第 {pos} 位")


if __name__ == "__main__":
    train()