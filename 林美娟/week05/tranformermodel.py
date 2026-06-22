#训练基于transformer的单向语言模型，并完成文本生成。训练文本基于corpus.txt文件
"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
    python language_model.py --model transformer --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from rich import print


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        #过滤掉不在词表中的字符
        ids = [char2idx[c] for c in text if c in char2idx]
        if len(ids) < seq_len + 1:
            raise ValueError("文本长度必须大于 seq_len + 1")
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Self Attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop(attn_out))
        # FFN
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x

class CorrectedLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads=8, ff_dim=1024, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        e = self.embed(x)
        T = e.size(1)
        # 创建 Causal Mask: (T, T)
        # True means masked out (invisible)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        out = e
        for block in self.blocks:
            out = block(out, mask)
        
        logits = self.fc(out)
        return logits
'''
class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads=8, ff=1024,dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.PositionalEncoding(embed_dim, dropout)
        # 使用 TransformerDecoderLayer 构建单向模型，或者手动实现 Causal Attention
        # 这里为了简单且正确，我们使用 nn.TransformerDecoder 
        # 注意：TransformerDecoder 需要 tgt_mask
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    

    def forward(self, x):
        e = self.embed(x)*math.sqrt(self.embed_dim)
        e = self.pos_encoder(e)
        T = e.size(1)
        causal_mask = torch.tril(torch.ones(T, T)).to(e.device)  # (T, T)
        out = self.transformer_decoder(e, e, tgt_mask=causal_mask)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits
'''

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)# 梯度裁剪，防止 Transformer 爆炸
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return float('inf'), float('inf')  # 避免除以零
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=[ "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    #parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--num_heads",  type=int,   default=8)
    parser.add_argument("--ff",         type=int,   default=1024)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="D:\\Users\\hp\\anaconda3\\envs\\py312\\dl-learning\\test\\*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")


    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    split_idx = int(len(text) * (1 - args.val_ratio))
    train_text = text[:split_idx]
    val_text   = text[split_idx:]

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}, Train chars: {len(train_text)}, Val chars: {len(val_text)}",)

    try:
        train_ds = CharDataset(train_text, char2idx, args.seq_len)
        val_ds   = CharDataset(val_text,   char2idx, args.seq_len)
    except Exception as e:
        print(f"数据集创建失败: {e}")
        return  
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    lines = text.splitlines()
    print(f"语料行数: {len(lines):,}")


    # 模型
    model = CorrectedLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff,
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
