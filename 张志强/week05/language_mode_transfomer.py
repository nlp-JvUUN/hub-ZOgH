
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    # 去重排序
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        # 构建索引
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        # Token Embedding
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # pos Embedding  nn.Parameter（可学习的位置编码）
        # torch.randn 生成一个标准正态分布（均值0，方差1）的随机tensor  参数1:batch维度 512:最大长度  embed_dim: 每个位置的维度  也就是为512个位置，每个位置准备一个embed_dim维的可学习向量
        # nn.Parameter(torch.randn(1, 512, embed_dim))这句代码也可以改为用Embedding方式:
        # self.pos_embed = nn.Embedding(512, embed_dim)
        # positions = torch.arange(seq_len).unsqueeze(0).to(x.device)
        # 最后 x = x + self.pos_embed(positions)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim))
        self.segment_embed = nn.Embedding(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # 模型维度（输入/输出特征维度）
            nhead=4,  # 注意力头数（将embed_dim分成4个头） bert里默认是12个头  该小一点
            dim_feedforward=128,  # FFN 神经元
            dropout=dropout,  # Dropout比率（防止过拟合）
            batch_first=True,  # 输入形状为 (batch, seq, dim)，而非 (seq, batch, dim
            activation="gelu"  # 激活函数（GELU比ReLU更平滑）
        )
        # 2层 Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    # 生成下三角 Mask（Attention Mask）
    def attention_mask(self, sz, device):
        mask = torch.triu(
            torch.ones(sz, sz, device=device),
            diagonal=1
        ).bool()

        return mask

    def forward(self, x):
        B, seq_len = x.size()
        e = self.embed(x) + self.pos_embed[:, :seq_len, :]
        e = self.drop(e)
        # 下三角 Attention Mask
        mask = self.attention_mask(seq_len, x.device)
        out = self.transformer(
            e,
            mask=mask
        )
        logits = self.fc(self.drop(out))  # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    """
    主函数：解析命令行参数，准备数据，训练和评估语言模型。
    
    命令行参数:
        --model:      模型类型，支持 'rnn' 或 'lstm'（默认: lstm）
        --epochs:     训练轮数（默认: 20）
        --seq_len:    序列长度（默认: 64）
        --batch_size: 批次大小（默认: 128）
        --embed_dim:  词嵌入维度（默认: 128）
        --hidden_dim: 隐藏层维度（默认: 256）
        --num_layers: RNN/LSTM 层数（默认: 2）
        --dropout:    Dropout 比率（默认: 0.3）
        --lr:         学习率（默认: 1e-3）
        --val_ratio:  验证集比例（默认: 0.05）
        --corpus:     语料文件路径模式（默认: "*.txt"）
        --save:       最佳模型保存路径（默认: "best_model.pt"）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model_trans.pt")
    args = parser.parse_args()

    # 该代码用于自动选择计算设备。它检测系统是否可用CUDA（GPU加速），若可用则使用"cuda”，否则回退到"cpu"。这确保模型能在性能最优的硬件上运行，同时保证在无GPU环境下的兼容性
    # Mac 并不是 CUDA。  应该使用mps
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")
    # 按行分割文本  返回list ['1111','222','xxx']
    lines = text.splitlines()
    # 打乱顺序
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    # 训练集
    train_text = "\n".join(lines[:split])
    # 验证集
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
