"""
基于 Transformer 的单向（因果/decoder-only）字符级语言模型。
自包含实现（MultiHeadAttention + FeedForward + TransformerDecoderLayer），
使用下三角 causal mask 确保单向注意力。训练完成后支持 top-k / top-p 采样文本生成。

用法:
    python transformer_lm.py --epochs 20
    python transformer_lm.py --generate --prompt "黄金" --temperature 0.8
"""

import math
import argparse
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════ Transformer 组件 ══════════════════════════

class MultiHeadAttention(nn.Module):
    """多头自注意力，手动计算 Q/K/V/Attention。"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 拆分为多头: (B, T, d_model) → (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # causal mask: True 的位置填 -inf（屏蔽）
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # 合并多头: (B, H, T, d_k) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.w_o(context)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    """两层全连接前馈网络: d_model → d_ff → d_model，GELU 激活。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """单向 Transformer 解码器层（Pre-LN 结构）。
    自动构造 causal mask，确保每个位置只能看到自身及之前的 token。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape

        # 构造 causal mask：上三角（不含对角线）为 True → 屏蔽
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # 子层 1: Self-Attention + 残差连接
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=causal_mask)
        x = self.dropout(x)
        x = residual + x

        # 子层 2: Feed-Forward + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


# ══════════════════════════ 数据 ══════════════════════════

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
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ══════════════════════════ 训练 / 评估 ══════════════════════════

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ══════════════════════════ 模型 ══════════════════════════

class TransformerLM(nn.Module):
    """GPT-style decoder-only Transformer 语言模型。

    结构: TokenEmbed + PosEmbed → Dropout → [TransformerDecoderLayer × N]
          → LayerNorm → Linear(→ vocab_size)
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """GPT-2 风格权重初始化。"""
        std = 0.02
        for name, p in self.named_parameters():
            if 'norm' in name:
                continue
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=std)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len, \
            f"序列长度 {T} 超过 max_seq_len {self.max_seq_len}"

        tok_emb = self.token_embed(x) * math.sqrt(self.d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)
        h = self.drop(tok_emb + pos_emb)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.fc(h)
        return logits


# ══════════════════════════ 文本生成 ══════════════════════════

@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device='cpu',
             max_new_tokens=200, temperature=1.0, top_k=0, top_p=0.9):
    """自回归文本生成，支持 top-k 和 top-p (nucleus) 采样。"""
    model.eval()

    ids = [char2idx.get(c, 0) for c in prompt]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = list(ids[0].tolist())

    for _ in range(max_new_tokens):
        ctx = ids[:, -model.max_seq_len:]
        logits = model(ctx)
        logits = logits[:, -1, :] / temperature

        # top-k 过滤
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[:, -1:]] = float('-inf')

        # top-p (nucleus) 过滤
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) >= top_p
            mask[:, 0] = False
            sorted_logits[mask] = float('-inf')
            logits = logits.new_zeros(logits.shape).scatter(
                dim=1, index=sorted_indices, src=sorted_logits
            )

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_id_val = next_id.item()
        generated.append(next_id_val)
        ids = torch.cat([ids, next_id], dim=1)

    return ''.join(idx2char.get(i, '?') for i in generated)


# ══════════════════════════ 主函数 ══════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Transformer 单向语言模型（decoder-only）训练"
    )
    parser.add_argument("--d_model",      type=int,   default=512)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=2048)
    parser.add_argument("--num_layers",   type=int,   default=6)
    parser.add_argument("--max_seq_len",  type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--seq_len",      type=int,   default=64)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--val_ratio",    type=float, default=0.05)
    parser.add_argument("--corpus",       default="corpus.txt")
    parser.add_argument("--save",         default="best_transformer.pt")
    parser.add_argument("--generate",     action="store_true",
                        help="训练完成后进入交互式文本生成")
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_k",        type=int,   default=0,
                        help="top-k 采样 (0=禁用)")
    parser.add_argument("--top_p",        type=float, default=0.9,
                        help="top-p / nucleus 采样 (0=禁用)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer (decoder-only)")
    print(f"config: d_model={args.d_model}  num_heads={args.num_heads}  "
          f"d_ff={args.d_ff}  num_layers={args.num_layers}")

    # ── 数据准备 ──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # ── 模型 ──
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "vocab_size": vocab_size,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "d_ff": args.d_ff,
                    "num_layers": args.num_layers,
                    "max_seq_len": args.max_seq_len,
                },
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # ── 文本生成演示 ──
    if args.generate:
        print(f"\n{'=' * 50}")
        print("文本生成演示（输入 q 退出）")
        print(f"采样策略: top_k={args.top_k}, top_p={args.top_p}, "
              f"temperature={args.temperature}")
        print(f"{'=' * 50}")
        while True:
            prompt = input("\n请输入开头文字: ").strip()
            if prompt.lower() == 'q':
                break
            if not prompt:
                prompt = "中国"
            result = generate(
                model, prompt, char2idx, idx2char, device=device,
                max_new_tokens=200, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
            )
            print(f"\n生成结果:\n{result}")


if __name__ == "__main__":
    main()
