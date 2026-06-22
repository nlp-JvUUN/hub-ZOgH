import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 44
N_SAMPLES   = 4000
MAXLEN      =5 
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 0.001 
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

def make_you():
    bg_chars = "我他她它好坏大小多少高低上下左右前后里外有无在的得地了着过吗呢啊吧呀喔哈这那么些个"
    target_pos = random.randint(0, 4)
    sentence_chars = []
    for i in range(5):
        if i == target_pos:
            sentence_chars.append("你")
        else:
            sentence_chars.append(random.choice(bg_chars))
    sent = "".join(sentence_chars)
    return sent, target_pos

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_you())
    random.shuffle(data)
    return data

def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for word in list(sent):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    if len(ids) < maxlen:
        ids = ids + [0] * (maxlen - len(ids))
    else:
        ids = ids[:maxlen]
    return ids

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

# ─── 4. 模型定义 ──────────────────────────────────
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN 的 batch_first=True 表示输入输出格式为 (batch, seq_len, hidden_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5) # 输出5个类别的原始分数 (logits)

    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)
        # out: (B, L, hidden_dim), h_n: (1, B, hidden_dim)
        out, h_n = self.rnn(embed) 
        # 取最后一个时间步的输出作为句子特征
        pooled = out[:, -1, :] 
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled) # 输出 (B, 5)，不要加任何激活函数！
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X) # 得到原始分数
            pred = torch.argmax(logits, dim=1) # 取分数最高的类别索引
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
           
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你喜欢这家店',      
        '这上边有你名字',       
        '我爱你你爱我',           
        '我有他和你的照片',         
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = torch.softmax(logits, dim=1) 
            pred_class = torch.argmax(logits, dim=1).item()
            
            print(f"输入：{sent}")
            print(f"预测“你”在第：{pred_class} 位")
            print(f"各类别概率(0-4位)：{['%.2f' % p for p in prob[0].tolist()]}")
            print("-" * 20)

if __name__ == '__main__':
    train()
  --- 推理示例 ---
输入：你喜欢这家店
预测“你”在第：0 位
各类别概率(0-4位)：['1.00', '0.00', '0.00', '0.00', '0.00']
--------------------
输入：这上边有你名字
预测“你”在第：4 位
各类别概率(0-4位)：['0.00', '0.00', '0.00', '0.00', '1.00']
--------------------
输入：我爱你你爱我
预测“你”在第：2 位
各类别概率(0-4位)：['0.00', '0.00', '0.67', '0.32', '0.00']
--------------------
输入：我有他和你的照片
预测“你”在第：4 位
各类别概率(0-4位)：['0.00', '0.00', '0.00', '0.00', '1.00']
