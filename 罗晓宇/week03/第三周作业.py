"""
中文句子关键词分类 —— 简单 LSTM 版本

任务：对一个任意包含“你”字的五个字的文本进行分类，“你”在第几位，就属于第几类
模型：Embedding → LSTM → 取最后隐藏状态 → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
WORDS = ['三', '上', '下', '不', '业', '个', '久', '了', '交', '产', '人', '今', '他', '任', '会', 
         '伞', '体', '作', '便', '做', '公', '冒', '出', '分', '到', '务', '十', '午', '厅', '高',
         '又', '吗', '吧', '呢', '和', '品', '在', '堵', '境', '多', '天', '她', '季', '它', '家', 
         '容', '小', '少', '就', '工', '市', '布', '带', '常', '平', '店', '度', '开', '影', '很', 
         '得', '忘', '态', '情', '感', '我', '换', '排', '效', '方', '时', '易', '是', '晚', '最', 
         '有', '服', '来', '极', '次', '款', '比', '气', '沉', '没', '洁', '淡', '点', '物', '特', 
         '独', '环', '电', '的', '真', '程', '简', '系', '繁', '结', '统', '置', '耽', '舒', '节', 
         '要', '觉', '解', '计', '让', '议', '设', '误', '说', '课', '账', '购', '超', '路', '车', 
         '较', '近', '还', '这', '适', '道', '部', '都', '里', '重', '钟', '铺', '门', '间', '队', 
         '阴', '雨', '非', '题', '餐', '验']

def make_sample():
    sent = random.choices(WORDS, k=5)
    pos  = random.randint(0, 4)
    sent[pos] = '你'
    return ''.join(sent), pos

def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
    random.shuffle(data) # 打乱顺序
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1, '你': 2}  # 预留0给PAD，1给UNK，2给“你”
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen] # 截断
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
    中文关键词分类器（LSTM + MaxPooling 版）
    架构：Embedding → LSTM → MaxPool → BN → Dropout → Linear → Softmax → (CrossEntropyLoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)  # 5个类别

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.lstm(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        out = self.dropout(self.bn(pooled)) # 无论train还是eval都应用bn（使用running stats），dropout自动在eval时关闭
        out = self.fc(out)                  # (B, 5)  输出logits
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits  = model(X)  # 模型输出logits
            probs   = torch.softmax(logits, dim=1)  # 转换为概率（对应模型文档中的Softmax）
            pred    = probs.argmax(dim=1)  # 对概率做argmax
            correct += (pred == y).sum().item()  # y已是long类型
            total   += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN(vocab_size=len(vocab))
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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '这款产品你',
        '今你天气有',
        '服务太你了',
        '你等了很久',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids    = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)  # (1, 5)
            label  = logits.argmax(dim=1).item()  # 沿类别维度找最大值
            prob   = torch.softmax(logits, dim=1)  # 转换为概率
            print(f"  [{label}({prob[0, label]:.2f})]  {sent}")


if __name__ == '__main__':
    train()
