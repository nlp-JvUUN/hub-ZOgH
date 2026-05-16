"""
Transformer Encoder
结构:
    Embedding + PositionalEncoding
    N x [LayerNorm → MultiHeadAttention → 残差 → LayerNorm → FFN → 残差]
    最终 LayerNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────── 1. 位置编码(sin/cos 固定) ───────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                     # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


# ─────────── 2. 多头注意力 ───────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        H, Dk = self.num_heads, self.d_k

        # 投影并拆头: (B, T, d_model) -> (B, H, T, Dk)
        q = self.W_q(x).view(B, T, H, Dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, Dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, Dk).transpose(1, 2)

        # 缩放点积: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(Dk)
        attn = F.softmax(scores, dim=-1)

        # 加权求和: (B, H, T, Dk)
        out = attn @ v

        # 合并头: (B, H, T, Dk) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_o(out)


# ─────────── 3. 前馈网络 (FFN) ───────────

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ─────────── 4. 单个 Encoder Block (Pre-LN) ───────────

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN: 先 LN 再进子层,残差直通
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────── 5. 完整 Transformer Encoder ───────────

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)   # Pre-LN 在最后再做一次 LN
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T) 整数 token id
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)                           # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)                        # (B, T, d_model)
        return x


# ─────────── 6. 主函数:模拟数据,打印形状 ───────────

def main():
    torch.manual_seed(0)

    # 超参
    B, T = 2, 10           # batch=2,序列长度=10
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )

    # 随机输入:整数 token id
    x = torch.randint(0, vocab_size, (B, T))
    print(f"输入形状: {x.shape}   dtype: {x.dtype}")
    print(f"输入样例(第 0 条): {x[0].tolist()}")

    # 前向
    y = model(x)
    print(f"输出形状: {y.shape}   dtype: {y.dtype}")
    print(f"输出第 0 条第 0 个 token 的前 5 维: {y[0, 0, :5].tolist()}")

    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total:,}")


if __name__ == "__main__":
    main()
