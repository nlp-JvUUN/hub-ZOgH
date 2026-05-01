"""
一个以文本为输入的五分类任务:

任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
优化：Adam (lr=1e-3)   损失：MSELoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5     # 五分类任务

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 生成5字文本的字符池
CHAR_POOL = ['好', '棒', '赞', '喜', '欢', '满', '意', '的', '是', '我',
             '这', '款', '很', '真', '太', '了', '都', '也', '还', '又',
             '在', '上', '下', '左', '右', '前', '后', '大', '小', '多']

def generate_5char_sentence(position):
    """生成包含"你"字的5字文本，"你"在指定位置(1-5)"""
    assert 1 <= position <= 5, "位置必须在1-5之间"
    chars = []
    for i in range(5):
        if i + 1 == position:
            chars.append('你')
        else:
            char = random.choice(CHAR_POOL)
            while char == '你':
                char = random.choice(CHAR_POOL)
            chars.append(char)
    return ''.join(chars)

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // NUM_CLASSES):
        for pos in range(1, NUM_CLASSES + 1):
            sent = generate_5char_sentence(pos)
            label = pos - 1
            data.append((sent, label))
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
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


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文位置分类器（RNN + 最后隐藏状态版）
    架构：Embedding → RNN → 取最后隐藏状态 → BN → Dropout → Linear → (CrossEntropyLoss)
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)  # 输出5个类别

    def forward(self, x):
        # x: (batch, seq_len=5)
        x_embed = self.embedding(x)  # (B, 5, embed_dim)
        output, hidden = self.rnn(x_embed)  # output: (B,5,hidden_dim); hidden: (1,B,hidden_dim)

        # 取最后隐藏状态 (B, hidden_dim)
        last_hidden = hidden.squeeze(0)  # 去除第一个维度(1,B,hidden_dim) → (B,hidden_dim)

        pooled = self.dropout(self.bn(last_hidden))
        out = self.fc(pooled)  # (B, NUM_CLASSES)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader, return_preds=False):
    model.eval()
    correct = total = 0
    y_list = []
    pred_list = []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)  # (B, NUM_CLASSES)
            pred = torch.argmax(logits, dim=1)  # 取概率最大的类别
            correct += (pred == y).sum().item()
            total += len(y)
            if return_preds:
                y_list.extend(y.numpy())
                pred_list.extend(pred.numpy())
    if return_preds:
        return correct / total, y_list, pred_list
    return correct / total

def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    print(f"数据集...{data}")
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}，类别数：{NUM_CLASSES}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 初始化记录训练过程数据的列表
    train_losses = []
    val_accuracies = []

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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # 记录每个epoch的损失和准确率
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # 获取预测值与真实值用于可视化
    val_acc, y_true, y_pred = evaluate(model, val_loader, return_preds=True)
    Y = [y + 1 for y in y_true]  # 真实位置 (1-5)
    Yp = [p + 1 for p in y_pred]  # 预测位置 (1-5)
    X = list(range(len(Y)))  # 样本索引

    # 预测值与真实值比对数据分布
    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, color="red", label="真实位置", alpha=0.6, s=30)
    plt.scatter(X, Yp, color="blue", label="预测位置", alpha=0.6, s=30)
    plt.xlabel("样本索引")
    plt.ylabel("位置 (1-5)")
    plt.title("真实位置与预测位置对比")
    plt.legend()
    plt.show()

    # 绘制训练损失和验证准确率曲线
    plt.figure(figsize=(16, 8))


    print("\n--- 推理示例 ---")
    model.eval()
    # 测试5个位置的示例（确保是5字文本且"你"在对应位置）
    test_sents = [
        "你好世界啊",  # "你"在第1位 → 类别0
        "好你世界啊",  # "你"在第2位 → 类别1
        "好世你界啊",  # "你"在第3位 → 类别2
        "好世界你啊",  # "你"在第4位 → 类别3
        "好世界啊你",  # "你"在第5位 → 类别4
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred = torch.argmax(logits, dim=1).item()
            print(f"  [预测位置：{pred+1} (类别{pred})]  {sent}")


if __name__ == '__main__':
    train()
