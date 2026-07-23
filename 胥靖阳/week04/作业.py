# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/5/14 18:05  
@脚本说明 : 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性变换并拆分为多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 应用掩码（若提供）
        if mask is not None:
            # mask shape: (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # 6. 合并多头并线性变换
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层（Pre-LN 结构）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: 先归一化再计算子层，残差连接在最后
        # 多头自注意力子层
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout1(x)
        x = residual + x

        # 前馈网络子层
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = residual + x

        return x


# ========== 示例用法 ==========
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 创建模型层
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

    # 随机输入 (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # 可选掩码 (这里使用 padding mask，形状 (batch, 1, 1, seq_len))
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0   # 最后两个位置 padding

    # 前向传播
    output = encoder_layer(x, mask=mask)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("Transformer 编码器层运行成功！")