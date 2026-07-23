"""
面试版 Transformer Encoder
核心：Multi-Head Self-Attention / FFN / 残差 + LN / 堆叠
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head, dropout=0.3):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head  # d_k = d_v 切分的每个头的容量
        self.qkv = nn.Linear(hidden, hidden * 3)   # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # [B, T, 3H] -> 3 个 [B, n_head, T, d_k]
        # 同时需要把 n_head和T换一下位置
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2) # [B, n_head, T, d_k]
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # scaled dot-product
        # scores[b, h, i, j] 表示 batch b、头 h 中，位置 i 对位置 j 的"关注程度"。除以 sqrt(d_k) 是为了防止内积过大导致 softmax 梯度饱和。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k) # [B, n_head, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # 将 mask 为 0 的位置的得分替换为 -1e9（近似负无穷）
        attn = F.softmax(scores, dim=-1)
        
        # if y is None: # training TODO 推理时需要关闭 只加入一次dropout
        attn = self.dropout(attn) # nn.Dropout 本身就会根据 model.train()/eval() 自动开关

        out = attn @ v                              # [B, n_head, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head, dropout)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))        # 残差 + LN
        x = self.dropout(x)
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12, ff=3072, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff, dropout) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    model = TransformerEncoder(hidden=512, n_layer=6, n_head=8, ff=1024)
    x = torch.randn(2, 16, 512)        # [B, T, H]
    print(model(x).shape)              # [2, 16, 512]
