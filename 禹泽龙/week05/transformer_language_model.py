"""
基于 Transformer Encoder 的单向字符级语言模型
==============================================
架构说明：
    - 使用 TransformerEncoder（来自 transformer_interview.py）作为核心
    - 单向语言模型：通过 Casual Mask 遮蔽未来位置
    - 支持训练和文本生成

关键设计：
    1. Causal Mask：确保每个位置只能看到当前及之前的字符
    2. 位置编码：Transformer 本身不感知位置，需要添加
    3. 训练：和 RNN 语言模型一样的 next-token 预测任务
    4. 生成：自回归方式，逐字符生成
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 导入 Transformer 组件（假设和本文件同一目录）
from transformer_interview import TransformerEncoder


# ═══════════════════════════════════════════════════════════════
# 位置编码：为 Transformer 添加位置感知能力
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    经典的三角函数位置编码
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    原理：不同位置通过不同频率的正弦/余弦组合来区分，
          模型可以学习到相对位置关系（因为 sin/cos 具有线性变换特性）
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 预先计算位置编码 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数维度
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 便于广播
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        将位置编码加到输入上（加法而非拼接，不增加参数量）
        """
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════
# 文本处理工具
# ═══════════════════════════════════════════════════════════════

def load_corpus(pattern="*.txt"):
    """加载目录下所有 txt 文件并拼接为一个字符串"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """从文本构建字符到索引的映射"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级数据集
    输入：长度为 seq_len 的字符序列
    目标：同一个序列向后偏移一位（预测下一个字符）
    """
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        # 过滤掉不在词表中的字符（如果有的话）
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 最后 seq_len 个字符不能作为起点（没有完整目标）
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]       # 输入
        y = self.data[idx + 1: idx + self.seq_len + 1]  # 目标（向后偏移一位）
        return x, y


# ═══════════════════════════════════════════════════════════════
# 核心模型：Transformer 语言模型
# ═══════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """
    基于 Transformer Encoder 的单向语言模型

    关键点：
    1. 词嵌入 + 位置编码 → 送入 Transformer Encoder
    2. causal_mask 确保单向（不能看到未来）
    3. 输出层预测下一个 token
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, n_head, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        # 使用传入的 hidden_dim 作为 Transformer 的 hidden size
        self.transformer = TransformerEncoder(
            hidden=hidden_dim,
            n_layer=num_layers,
            n_head=n_head,
            ff=hidden_dim * 4  # FFN 中间层通常设为 hidden * 4
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # 权重绑定：词嵌入和输出层共享（减少参数量，提高泛化）
        self.fc.weight = self.embed.weight

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len]  字符索引
        mask: 可选，用于遮蔽某些位置
        """
        # 1. 词嵌入 + 位置编码
        e = self.embed(x) * math.sqrt(self.embed.embedding_dim)  # 缩放防止方差变小
        e = self.pos_enc(e)
        e = self.drop(e)

        # 2. 生成 causal mask（关键：遮蔽未来位置）
        seq_len = x.size(1)
        # 下三角矩阵，对角线及以下为 1（可见），以上为 0（遮蔽）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # 将 mask 转为 Transformer 期望的格式
        # mask=0 表示该位置需要被遮蔽
        if mask is None:
            mask = causal_mask
        else:
            mask = mask & causal_mask

        # 3. Transformer forward
        out = self.transformer(e, mask=mask)  # [batch, seq_len, hidden_dim]

        # 4. 预测下一个字符
        logits = self.fc(out)  # [batch, seq_len, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, start_ids, idx2char, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        自回归文本生成

        参数：
            start_ids: 起始字符索引序列 [seq_len]
            idx2char: 索引到字符的映射
            max_new_tokens: 最大生成字符数
            temperature: 温度参数（越高越随机，越低越确定）
            top_k: 限制只从 top-k 中采样

        流程：
            1. 将起始序列输入模型，获取下一个字符的预测分布
            2. 根据温度采样一个字符
            3. 将该字符追加到序列末尾
            4. 重复直到达到最大长度
        """
        self.eval()
        device = next(self.parameters()).device
        start_ids = start_ids.to(device)

        # 自回归生成
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到最后 seq_len 个字符
            seq_len = self.pos_enc.pe.size(1)
            input_ids = start_ids[:, -seq_len:] if start_ids.size(1) > seq_len else start_ids

            # 前向传播获取 logits
            logits = self.forward(input_ids)  # [1, cur_len, vocab_size]
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置的预测

            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 转为概率分布并采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 追加到序列
            start_ids = torch.cat([start_ids, next_token], dim=1)

        # 索引转字符
        generated_text = ''.join([idx2char[i.item()] for i in start_ids[0]])
        return generated_text


# ═══════════════════════════════════════════════════════════════
# 训练和评估
# ═══════════════════════════════════════════════════════════════

def compute_ppl(loss):
    """从交叉熵损失计算困惑度"""
    return math.exp(loss)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    运行一个训练/验证 epoch

    模型输出 [batch, seq_len, vocab_size]
    目标 [batch, seq_len]
    展平后计算交叉熵损失
    """
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
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = compute_ppl(avg_loss)
    return avg_loss, ppl


# ═══════════════════════════════════════════════════════════════
# 文本生成演示
# ═══════════════════════════════════════════════════════════════

def demo_generation(model, char2idx, idx2char, device, prompt="", num_chars=200):
    """
    演示如何使用模型生成文本
    """
    if prompt:
        # 将提示语转为索引
        start_ids = torch.tensor([[char2idx.get(c, 0) for c in prompt]], dtype=torch.long)
    else:
        # 随机选一个起始字符
        start_ids = torch.tensor([[random.choice(list(char2idx.values()))]], dtype=torch.long)

    print(f"\n{'='*60}")
    print(f"生成示例（起始: \"{prompt}\"）")
    print(f"{'='*60}")
    print(f"模型生成: ", end="")

    generated = model.generate(
        start_ids,
        idx2char,
        max_new_tokens=num_chars,
        temperature=0.8,  # 较低温度，生成相对确定的文本
        top_k=40
    )

    # 只打印新生成的部分
    if prompt:
        print(generated[len(prompt):])
    else:
        print(generated)
    print()


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transformer 语言模型训练")
    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=512, help="词嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Transformer hidden 维度")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1)
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64, help="输入序列长度")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    # 数据和保存
    parser.add_argument("--corpus", default="*.txt", help="语料文件路径")
    parser.add_argument("--save", default="transformer_lm.pt", help="模型保存路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"Transformer 语言模型")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"模型配置: embed={args.embed_dim}, hidden={args.hidden_dim}, "
          f"layers={args.num_layers}, heads={args.n_head}")

    # ── 数据准备 ──────────────────────────────────────────────
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按行 shuffle 后划分训练/验证集（保持行的完整性）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ── 模型初始化 ──────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ────────────────────────────────────────────────
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()

        # 保存最佳模型
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

    # ── 生成演示 ────────────────────────────────────────────────
    # 加载最佳模型进行生成
    checkpoint = torch.load(args.save, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # 演示几种不同温度的生成
    for temp in [0.5, 0.8, 1.0]:
        print(f"\n[温度={temp}]")
        demo_generation(model, char2idx, idx2char, device,
                        prompt="今天", num_chars=100)


if __name__ == "__main__":
    main()
