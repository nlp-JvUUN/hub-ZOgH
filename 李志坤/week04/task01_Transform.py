"""
实现 Transformer Encoder 组件编码
内容包括：多头自注意力机制、前馈神经网络、残差连接、层归一化
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model  = d_model
        self.num_heads = num_heads
        self.d_k      = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        Q = self.split_heads(self.W_q(x), batch_size)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output  = torch.matmul(attn_weights, V)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        return self.W_o(attn_output)


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 单层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class TransformerEncoder(nn.Module):
    """完整的 Transformer Encoder"""
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == '__main__':
    batch_size = 2
    seq_len    = 5
    d_model    = 768
    num_heads  = 12
    num_layers = 12
    d_ff       = 3072

    x = torch.randn(batch_size, seq_len, d_model)

    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    print(f"Encoder Layer 输出形状: {encoder_layer(x).shape}")

    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    print(f"Encoder 输出形状: {encoder(x).shape}")

    multi_head_attn = MultiHeadSelfAttention(d_model, num_heads)
    print(f"Multi-Head Attention 输出形状: {multi_head_attn(x).shape}")

    ffn = FeedForwardNetwork(d_model, d_ff)
    print(f"Feed Forward Network 输出形状: {ffn(x).shape}")
