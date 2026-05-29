"""
根据 language_model.py 改写语言模型，基于transformer实现

python 第五周作业.py --generate_only --load diwuzhouzuoye_transformer_lm_best.pt --prompt "人工智能"
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_CORPUS = r"D:\aipy\AI大模型培训部分\week5大语言模型初探_0517\corpus.txt"
DEFAULT_SAVE = r"D:\aipy\AI大模型培训部分\week5大语言模型初探_0517\diwuzhouzuoye_transformer_lm_best.pt"

# --------------------------- 数据 ---------------------------

def load_corpus(pattern=DEFAULT_CORPUS):
    # 支持 glob 通配符，便于一次加载一个或多个语料文件。
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)

def build_vocab(text):
    # 字符级语言模型：每个不同字符对应一个 token id。
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        # 将文本转换为 token id 序列，训练时用滑动窗口切成固定长度样本。
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # x 是当前片段，y 是整体右移一位后的“下一个字符”标签。
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# ─────────────────────────── 模型 ───────────────────────────
def build_causal_mask(seq_len, device):
    """下三角可见、上三角遮挡；True 表示该位置会被 attention mask 掉。"""
    lower_triangle_visible = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    return ~lower_triangle_visible

class TransformerLM(nn.Module):
    def __init__(self,vocab_size,embed_dim=128,hidden_dim=128,num_layers=3,num_heads=4,dropout=0.1,max_seq_len=64,):
        super().__init__()
        self.max_seq_len = max_seq_len
        # token embedding 表示字符本身，position embedding 表示字符在上下文中的位置。
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度 {seq_len} 超过 max_seq_len={self.max_seq_len}")

        # 为 batch 中每条样本构造相同的位置编号，再与字符向量相加。
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        h = self.token_embed(x) + self.position_embed(positions)
        h = self.drop(h)

        # 自回归生成只能看见当前位置及其左侧内容，不能提前看到未来字符。
        causal_mask = build_causal_mask(seq_len, x.device)
        out = self.encoder(h, mask=causal_mask)
        # 输出每个位置对词表中所有字符的预测分数。
        logits = self.fc(self.norm(out))
        return logits

# --------------------------- 训练 / 评估 ---------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # CrossEntropyLoss 需要二维 logits 和一维标签，所以把 batch/seq 两维展平。
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪可以降低训练初期梯度爆炸的风险。
            optimizer.step()

        # 按 token 数累计 loss，最后得到全量 token 的平均损失。
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) 
    return avg_loss, ppl

def set_seed(seed):
    # 固定随机种子，方便复现实验和生成结果。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------- 文本生成 ---------------------------
#temperature及top-p采样
def sample_top_p(logits, temperature=1.5, top_p=0.9):
    # temperature 越高越随机，越低越保守。
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    # nucleus sampling：只保留累计概率不超过 top_p 的高概率候选。
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove_mask = cumulative_probs > top_p
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(sorted_remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # 在筛选后的候选集中随机采样，并映射回原词表 id。
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    sampled_idx = sorted_indices.gather(-1, sampled_sorted_idx)
    return sampled_idx.squeeze(-1)        

@torch.no_grad()
def generate_text(
    model,
    char2idx,
    idx2char,
    device,
    prompt="语言模型",
    max_new_chars=30,
    temperature=1.5,
    top_p=0.9,
):
    model.eval()
    # 忽略词表外字符；如果 prompt 完全不可用，则随机选一个字符作为起点。
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]

    if not ids:
        first_idx = random.randrange(len(idx2char))
        ids = [first_idx]
        generated = idx2char[first_idx]
    else:
        generated = prompt

    for _ in range(max_new_chars):
        # 只保留模型最大上下文长度范围内的最近字符。
        context = ids[-model.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        # 使用最后一个位置的 logits 预测下一个字符。
        next_id = sample_top_p(logits, temperature=temperature, top_p=top_p).item()
        ids.append(next_id)
        generated += idx2char[next_id]

    return generated

# --------------------------- 主函数 ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--load", default=None)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="语言模型")
    parser.add_argument("--gen_len", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TransformerLM")

    if args.generate_only:
        load_path = args.load or args.save
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到模型文件: {load_path}")

        # checkpoint 中保存了模型参数、词表和训练时的结构配置。
        checkpoint = torch.load(load_path, map_location=device)
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        checkpoint_args = checkpoint.get("args", {})

        # 加载时必须使用和训练时一致的模型结构，否则 state_dict 无法匹配。
        model = TransformerLM(
            vocab_size=len(char2idx),
            embed_dim=checkpoint_args.get("embed_dim", args.embed_dim),
            hidden_dim=checkpoint_args.get("hidden_dim", args.hidden_dim),
            num_layers=checkpoint_args.get("num_layers", args.num_layers),
            num_heads=checkpoint_args.get("num_heads", args.num_heads),
            dropout=checkpoint_args.get("dropout", args.dropout),
            max_seq_len=checkpoint_args.get("seq_len", args.seq_len),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        print(f"已加载模型: {load_path}")
        print(f"采样策略: Temperature={args.temperature}, Top-P={args.top_p}")
        print("\n生成结果:")
        print(generate_text(
            model,
            char2idx,
            idx2char,
            device,
            prompt=args.prompt,
            max_new_chars=args.gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        ))
        return

    print(
        "config: "
        f"layers={args.num_layers}, hidden={args.hidden_dim}, "
        f"embed={args.embed_dim}, heads={args.num_heads}"
    )

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到或无法读取语料文件: {args.corpus}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    # 简单按行切分训练集和验证集，val_ratio 控制验证集比例。
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)
   
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    saved_model = False
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            # 只保存验证集 PPL 最低的模型，作为当前训练过程的 best checkpoint。
            best_val_ppl = va_ppl
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                args.save,
            )
            saved_model = True

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"采样策略: Temperature={args.temperature}, Top-P={args.top_p}")
    print("\n生成示例:")
    print(generate_text(
        model,
        char2idx,
        idx2char,
        device,
        prompt=args.prompt,
        max_new_chars=args.gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    ))


if __name__ == "__main__":
    main()






