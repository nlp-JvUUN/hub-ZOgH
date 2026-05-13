import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """多头注意力 - 用分开的投影层"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        # 分开投影（和面试版不同）
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.heads
        d = D // H
        
        # 投影并拆头
        q = self.q(x).view(B, T, H, d).transpose(1, 2)
        k = self.k(x).view(B, T, H, d).transpose(1, 2)
        v = self.v(x).view(B, T, H, d).transpose(1, 2)
        
        # 注意力
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # 合并
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class Block(nn.Module):
    """Transformer层 - Pre-Norm风格（和面试版Post-Norm不同）"""
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        # 前馈网络 - 用Conv1d实现（和面试版不同）
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),           # 面试版用GELU
            nn.Linear(ff_dim, dim),
        )
        
    def forward(self, x, mask=None):
        # Pre-Norm: 先norm再attention（面试版是后norm）
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """堆叠多层"""
    def __init__(self, dim=512, layers=6, heads=8, ff_dim=2048):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, heads, ff_dim) for _ in range(layers)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


# 测试
if __name__ == "__main__":
    model = TransformerEncoder(dim=512, layers=6, heads=8, ff_dim=2048)
    x = torch.randn(2, 16, 512)
    print(f"输入: {x.shape} -> 输出: {model(x).shape}")