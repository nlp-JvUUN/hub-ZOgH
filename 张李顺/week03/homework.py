"""
任务：5字样本分类，样本必含"你"，"你"在第几位就属于第几类（0~4）
模型：CNN + LSTM 混合 + LayerNorm
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
class Config:
    # 数据相关
    SEQ_LEN       = 5         # 序列长度（5个字）
    NUM_CLASSES   = 5         # 类别数（"你"的位置 0~4）
    TRAIN_SIZE    = 5000      # 训练样本数
    TEST_SIZE     = 1000      # 测试样本数
    VOCAB_CHARS   = "我他她它们好坏大小高低天地人山水火木金土风云日月星" \
                    "光明暗夜白黑红蓝绿黄上下左右前后东西南北中"  # 备选字（不含"你"）

    # 模型相关
    EMBED_DIM     = 32        # 词嵌入维度
    CNN_CHANNELS  = 64        # CNN 输出通道数
    CNN_KERNEL    = 3         # CNN 卷积核大小
    LSTM_HIDDEN   = 64        # LSTM 隐藏层维度
    LSTM_LAYERS   = 1         # LSTM 层数
    DROPOUT       = 0.2

    # 训练相关
    BATCH_SIZE    = 64
    EPOCHS        = 10
    LR            = 1e-3
    SEED          = 42
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ============================================================
# 词表构建
# ============================================================
# 把"你"和备选字组合成词表，每个字一个 id；0 留给 padding（这里没用到，习惯保留）
chars = ["<pad>", "你"] + list(cfg.VOCAB_CHARS)
char2id = {c: i for i, c in enumerate(chars)}
VOCAB_SIZE = len(chars)

# ============================================================
# 数据生成：随机选 5 个字，强制把"你"放到某一位
# ============================================================
def gen_sample():
    pos = random.randint(0, cfg.SEQ_LEN - 1)        # "你"的位置 = 标签
    others = random.choices(cfg.VOCAB_CHARS, k=cfg.SEQ_LEN - 1)
    seq = others[:pos] + ["你"] + others[pos:]
    ids = [char2id[c] for c in seq]
    return ids, pos

class CharDataset(Dataset):
    def __init__(self, n):
        self.data = [gen_sample() for _ in range(n)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ============================================================
# 模型：Embedding -> CNN -> LayerNorm -> LSTM -> LayerNorm -> FC
# ============================================================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.EMBED_DIM, padding_idx=0)

        # CNN：在序列维度上做一维卷积，提取局部 n-gram 特征
        self.conv = nn.Conv1d(
            in_channels=cfg.EMBED_DIM,
            out_channels=cfg.CNN_CHANNELS,
            kernel_size=cfg.CNN_KERNEL,
            padding=cfg.CNN_KERNEL // 2   # 保持序列长度不变
        )
        self.ln_cnn = nn.LayerNorm(cfg.CNN_CHANNELS)   # 对每个时间步的通道做归一化

        # LSTM：捕捉序列上下文，双向能更好定位"你"
        self.lstm = nn.LSTM(
            input_size=cfg.CNN_CHANNELS,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.ln_lstm = nn.LayerNorm(cfg.LSTM_HIDDEN * 2)

        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.fc = nn.Linear(cfg.LSTM_HIDDEN * 2, cfg.NUM_CLASSES)

    def forward(self, x):
        # x: (B, L)
        e = self.embed(x)                       # (B, L, E)
        c = self.conv(e.transpose(1, 2))        # (B, C, L)
        c = torch.relu(c).transpose(1, 2)       # (B, L, C)
        c = self.ln_cnn(c)                      # 沿通道维归一化

        out, _ = self.lstm(c)                   # (B, L, 2H)
        out = self.ln_lstm(out)

        # 池化：把整个序列压成一个向量再分类
        # 用 max-pool 比 mean 更敏感于"你"那一位的强响应
        pooled, _ = out.max(dim=1)              # (B, 2H)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# ============================================================
# 训练 & 评估
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    train_loader = DataLoader(CharDataset(cfg.TRAIN_SIZE), batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(CharDataset(cfg.TEST_SIZE),  batch_size=cfg.BATCH_SIZE)

    model = CNN_LSTM().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device: {cfg.DEVICE} | Vocab: {VOCAB_SIZE} | Params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
        acc = evaluate(model, test_loader)
        print(f"Epoch {ep:02d} | loss={total_loss/cfg.TRAIN_SIZE:.4f} | test_acc={acc:.4f}")

    # 抽几个样本看看预测
    print("\n--- 抽样验证 ---")
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            ids, label = gen_sample()
            x = torch.tensor([ids], device=cfg.DEVICE)
            pred = model(x).argmax(dim=1).item()
            text = "".join(chars[i] for i in ids)
            print(f"样本: {text} | 真实位置: {label} | 预测: {pred}")

if __name__ == "__main__":
    main()
