"""
训练基于transformer的单向语言模型，并完成文本生成。
1、训练模型 (作业1)
2、文本生成 (作业2)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import random
import math

# ===================== 数据 =====================
def load_corpus(path):
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        texts.append(f.read())
    return "".join(texts)

def build_vocab(data):
    texts = sorted(set(data))
    idx2char = {i:c for i, c in enumerate(texts)}
    char2idx = {c:i for i, c in idx2char.items()}
    return idx2char, char2idx

class TextDataSet(Dataset):
    def __init__(self, text, char2id, seq_len):
        super().__init__()
        self.seq_len = seq_len
        ids = [char2id[c] for c in text if c in char2id]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return (
            self.data[index: index + self.seq_len],
            self.data[index + 1: index + self.seq_len + 1]
        )

# ===================== 模型 =====================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, dim_feedforward, num_head, num_layers, dropout):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            activation=nn.functional.gelu,
            batch_first=True,
        )
        norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch, seq_len = x.shape
        we = self.word_embedding(x)     # (batch, seq_len, embed_dim)
        # 位置编码：生成 0..seq_len-1 的索引，并扩展 batch 维度
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pe = self.position_embedding(pos_ids) # (1, seq_len, embed_dim)
        embed_x = we + pe
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        transformer_x = self.transformer.forward(embed_x, mask=mask)   # (batch, seq_len, embed_dim)
        logits = self.output(transformer_x)                       # (B, T, vocab_size)
        return logits


# ===================== 训练、验证 =====================
def run_epoch(model, loader, criterion, optimizer, device, train=False):
    total_loss = 0.0
    total_tokens = 0
    if train:
        model.train()
    else:
        model.eval()
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


# ===================== 主函数 =====================
def main():
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--save", default="best_model.pt")
    args = parser.parse_args()

    # 设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 数据
    base_dir = r"/Users/wangqilin/个人/学习/week5大语言模型初探"
    path = os.path.join(base_dir, "corpus.txt")
    text = load_corpus(path)
    if not text:
        raise FileNotFoundError("语料不存在")
    
    # 构建词表
    idx2char, char2idx = build_vocab(text)
    vocab_size = len(idx2char)
    print(f"总词表:{vocab_size}")

    # 数据分割(训练集、验证集)
    texts = text.splitlines()
    random.shuffle(texts)
    split = int(len(texts) * (1 - args.val_ratio))
    train_text = "\n".join(texts[:split])
    val_text = "\n".join(texts[split:])

    train_data = TextDataSet(train_text, char2idx, args.seq_len)
    val_data = TextDataSet(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True)

    # 模型
    model = TransformerModel(
        vocab_size=vocab_size, 
        seq_len=args.seq_len,
        embed_dim=args.embed_dim, 
        dim_feedforward=args.ff_dim, 
        num_head=args.num_head, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数:{total_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练过程
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_loss, val_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {val_loss:>10.4f}  {val_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

if __name__ == "__main__":
    main()

