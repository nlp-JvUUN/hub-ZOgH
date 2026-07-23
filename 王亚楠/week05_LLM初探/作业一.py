"""
字符级语言模型 — Decoder-only Transformer 版本
支持自回归生成与四种解码策略（全部手动实现）

用法:
    python language_model.py --mode train --epochs 10
    python language_model.py --mode inference --prompt "春眠不觉晓" --strategy greedy
    python language_model.py --mode inference --prompt "春眠不觉晓" --strategy temperature_topk --temperature 0.8 --topk 50
    python language_model.py --mode inference --prompt "春眠不觉晓" --strategy temperature_topp --temperature 0.8 --topp 0.9
    python language_model.py --mode inference --prompt "春眠不觉晓" --strategy beam --beam_width 5
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ====================================================================
#  数据
# ====================================================================

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


# ====================================================================
#  Transformer 组件（手动实现）
# ====================================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                     # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.scale                   # (B, H, T, T)

        # 因果掩码：只允许看到当前及之前的 token
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                                                      # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)              # (B, T, d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """两层全连接前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    """单层 Transformer Decoder = 因果自注意力 + 前馈网络 (+ 残差 & LayerNorm)"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.self_attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-Only Transformer 语言模型"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 num_layers=1, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, T)
        x = self.embed(x) * math.sqrt(self.d_model)   # 缩放嵌入
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x)                               # (B, T, vocab_size)


# ====================================================================
#  训练 & 评估
# ====================================================================

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0

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


# ====================================================================
#  解码策略（全部手动实现）
# ====================================================================

@torch.no_grad()
def greedy_decode(model, char2idx, idx2char, prompt, max_len, device):
    """贪心解码：每步选择概率最高的 token"""
    model.eval()
    ids = [char2idx.get(c, 0) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(x)
        next_logits = logits[0, -1, :]
        next_token = torch.argmax(next_logits).item()
        ids.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

    return "".join(idx2char[t] for t in ids)


@torch.no_grad()
def temperature_topk_decode(model, char2idx, idx2char, prompt, max_len,
                            temperature, topk, device):
    """Temperature + Top-K 采样"""
    model.eval()
    ids = [char2idx.get(c, 0) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    vocab_size = len(char2idx)

    for _ in range(max_len):
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature

        if topk > 0 and topk < vocab_size:
            topk_vals, topk_idx = torch.topk(next_logits, topk)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered[topk_idx] = topk_vals
            probs = F.softmax(filtered, dim=-1)
        else:
            probs = F.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(probs, 1).item()
        ids.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

    return "".join(idx2char[t] for t in ids)


@torch.no_grad()
def temperature_topp_decode(model, char2idx, idx2char, prompt, max_len,
                            temperature, topp, device):
    """Temperature + Top-P (Nucleus) 采样"""
    model.eval()
    ids = [char2idx.get(c, 0) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)

        # 按概率从大到小排序
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # 找到累积概率第一次超过 topp 的位置，从那之后全部置零
        cutoff_mask = cumsum > topp
        cutoff_pos = cutoff_mask.nonzero(as_tuple=False)
        if len(cutoff_pos) > 0:
            first = cutoff_pos[0].item()
            sorted_probs[first + 1:] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

        chosen = torch.multinomial(sorted_probs, 1).item()
        next_token = sorted_idx[chosen].item()
        ids.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

    return "".join(idx2char[t] for t in ids)


@torch.no_grad()
def beam_search_decode(model, char2idx, idx2char, prompt, max_len,
                       beam_width, device):
    """束搜索"""
    model.eval()
    base_ids = [char2idx.get(c, 0) for c in prompt]

    # 每个 beam: (token_id 列表, 累积对数概率)
    beams = [(base_ids, 0.0)]

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            top_lp, top_idx = torch.topk(log_probs, beam_width)

            for lp, tok in zip(top_lp.tolist(), top_idx.tolist()):
                candidates.append((seq + [tok], score + lp))

        candidates.sort(key=lambda t: t[1], reverse=True)
        beams = candidates[:beam_width]

    best_seq, best_score = beams[0]
    return "".join(idx2char[t] for t in best_seq)


# ====================================================================
#  推理入口
# ====================================================================

DECODE_STRATEGIES = {
    "greedy":            greedy_decode,
    "temperature_topk":  temperature_topk_decode,
    "temperature_topp":  temperature_topp_decode,
    "beam":              beam_search_decode,
}


def run_inference(checkpoint_path, prompt, strategy, max_len,
                  temperature, topk, topp, beam_width, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    vocab_size = saved_args.get("vocab_size", len(char2idx))

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=saved_args.get("d_model", 128),
        n_heads=saved_args.get("n_heads", 4),
        d_ff=saved_args.get("d_ff", 512),
        num_layers=saved_args.get("num_layers", 1),
        dropout=saved_args.get("dropout", 0.1),
        max_len=saved_args.get("max_len", 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n模型已加载: {checkpoint_path}")
    print(f"策略: {strategy}  提示词: '{prompt}'\n")

    kwargs = dict(model=model, char2idx=char2idx, idx2char=idx2char,
                  prompt=prompt, max_len=max_len, device=device)
    if strategy == "temperature_topk":
        kwargs.update(temperature=temperature, topk=topk)
    elif strategy == "temperature_topp":
        kwargs.update(temperature=temperature, topp=topp)
    elif strategy == "beam":
        kwargs.update(beam_width=beam_width)

    result = DECODE_STRATEGIES[strategy](**kwargs)
    print(result)
    return result


# ====================================================================
#  主函数
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Decoder-Only Transformer 语言模型")
    parser.add_argument("--mode", default="train",
                        choices=["train", "inference"])

    # 模型结构
    parser.add_argument("--d_model",    type=int, default=128)
    parser.add_argument("--n_heads",    type=int, default=4)
    parser.add_argument("--d_ff",       type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--max_len",    type=int, default=512)

    # 训练
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="transformer_model.pt")

    # 推理
    parser.add_argument("--checkpoint",   default="transformer_model.pt")
    parser.add_argument("--prompt",       default="")
    parser.add_argument("--strategy",     default="greedy",
                        choices=["greedy", "temperature_topk", "temperature_topp", "beam"])
    parser.add_argument("--max_new",      type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--topk",         type=int,   default=50)
    parser.add_argument("--topp",         type=float, default=0.9)
    parser.add_argument("--beam_width",   type=int,   default=5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 推理模式 ----
    if args.mode == "inference":
        run_inference(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            strategy=args.strategy,
            max_len=args.max_new,
            temperature=args.temperature,
            topk=args.topk,
            topp=args.topp,
            beam_width=args.beam_width,
            device=device,
        )
        return

    # ---- 训练模式 ----
    print(f"device: {device}  model: Decoder-Only Transformer (layers={args.num_layers})")

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    args.vocab_size = vocab_size   # 一并存入 checkpoint 供推理使用

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    fmt = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}"
    print(f"\n{fmt}\n" + "-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer,
                                        device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  模型已保存至 {args.save}")


if __name__ == "__main__":
    main()
