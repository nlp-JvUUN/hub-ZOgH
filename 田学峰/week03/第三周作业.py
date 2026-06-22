"""
train_chinese_multiclass.py
中文文本多分类任务 —— RNN / LSTM 对比实验

任务：输入随机生成的中文句子，预测所属分类（体育、科技、娱乐、财经、教育）
模型1: Embedding → RNN → MaxPool → Linear → Softmax
模型2: Embedding → LSTM → MaxPool → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import torchm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5

random.seed(SEED)
torch.manual_seed(SEED)

CATEGORIES = ['体育', '科技', '娱乐', '财经', '教育']
SPORT_KEYWORDS = ['足球', '篮球', '比赛', '冠军', '运动员', '球队', '训练', '奥运', '金牌', '体育']
TECH_KEYWORDS  = ['手机', '电脑', '人工智能', '算法', '数据', '网络', '软件', '硬件', '智能', '科技']
ENTER_KEYWORDS = ['电影', '音乐', '明星', '演员', '导演', '综艺', '票房', '演唱会', '娱乐', '节目']
FIN_KEYWORDS   = ['股票', '投资', '银行', '经济', '市场', '基金', '汇率', '金融', '财富', '财经']
EDU_KEYWORDS   = ['学生', '学校', '老师', '考试', '学习', '教育', '课程', '大学', '知识', '校园']

TEMPLATES = [
    '今天的{}非常精彩，吸引了大量观众',
    '关于{}的最新消息引发了热议',
    '{}行业迎来了新的发展机遇',
    '专家表示{}将会改变我们的生活方式',
    '最近{}领域取得了重大突破',
]

def generate_sample(category, idx):
    if category == '体育':
        keywords = SPORT_KEYWORDS
    elif category == '科技':
        keywords = TECH_KEYWORDS
    elif category == '娱乐':
        keywords = ENTER_KEYWORDS
    elif category == '财经':
        keywords = FIN_KEYWORDS
    else:
        keywords = EDU_KEYWORDS

    template = random.choice(TEMPLATES)
    keyword = random.choice(keywords)
    text = template.format(keyword)

    if random.random() < 0.3:
        extra = random.choice(keywords)
        pos = random.randint(0, len(text))
        text = text[:pos] + extra + text[pos:]

    return text

def build_dataset(n=N_SAMPLES):
    samples_per_class = n // NUM_CLASSES
    data = []
    for cat in CATEGORIES:
        for i in range(samples_per_class):
            text = generate_sample(cat, i)
            data.append((text, cat))
    random.shuffle(data)
    return data

def build_vocab(data, min_freq=2):
    counter = Counter()
    for text, _ in data:
        counter.update(text)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for char, freq in counter.items():
        if freq >= min_freq:
            vocab[char] = len(vocab)
    return vocab

def encode(text, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in text]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, data, vocab, cat2idx):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [cat2idx[cat] for _, cat in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        e, _ = self.lstm(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total

def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"  Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"  {model_name} 最终验证准确率: {final_acc:.4f}")
    return final_acc

def main():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)

    cat2idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    idx2cat = {i: cat for cat, i in cat2idx.items()}

    print(f"  总样本数: {len(data)}")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  类别: {CATEGORIES}")

    vocab = build_vocab(data)
    print(f"  词表大小: {len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(
        TextDataset(train_data, vocab, cat2idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val_data, vocab, cat2idx),
        batch_size=BATCH_SIZE
    )

    print("\n" + "="*50)
    print("训练 RNN 模型...")
    print("="*50)
    rnn_model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"  RNN 参数量: {rnn_params:,}")
    rnn_acc = train_model(rnn_model, train_loader, val_loader, EPOCHS, LR, "RNN")

    print("\n" + "="*50)
    print("训练 LSTM 模型...")
    print("="*50)
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  LSTM 参数量: {lstm_params:,}")
    lstm_acc = train_model(lstm_model, train_loader, val_loader, EPOCHS, LR, "LSTM")

    print("\n" + "="*50)
    print("实验结果对比")
    print("="*50)
    print(f"  RNN  最终验证准确率: {rnn_acc:.4f}")
    print(f"  LSTM 最终验证准确率: {lstm_acc:.4f}")

    print("\n--- 推理示例 ---")
    lstm_model.eval()
    test_texts = [
        "今天的足球比赛非常精彩",
        "人工智能技术发展迅速",
        "新电影票房创新高",
        "股票市场投资机会",
        "学生考试取得好成绩",
    ]
    with torch.no_grad():
        for text in test_texts:
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
            outputs = lstm_model(ids)
            pred_idx = outputs.argmax(dim=1).item()
            probs = torch.softmax(outputs, dim=1)[0]
            print(f"  文本: {text}")
            print(f"  预测: {idx2cat[pred_idx]} (概率: {probs[pred_idx]:.3f})")
            print()

if __name__ == '__main__':
    main()
