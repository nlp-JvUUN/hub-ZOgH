# coding: utf-8
import os
import math

# 允许重复加载 OpenMP，避免部分 Windows 环境下运行 torch 时出现冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    # 多头自注意力模块
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()

        # hidden_size 必须能被头数整除，这样每个头才能平均分到一部分维度
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # Q、K、V 三个线性层，输入和输出维度都为 hidden_size
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        # 多头注意力结果拼接后，再通过一个输出线性层
        self.o = nn.Linear(hidden_size, hidden_size)

        # dropout 用于注意力权重，减少过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # x 的形状为 [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.size()

        # 先通过线性层得到 Q、K、V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 把 hidden_size 拆成 num_heads 个头
        # 结果形状变为 [batch_size, num_heads, seq_len, head_size]
        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)

        # 计算注意力分数：QK^T / sqrt(head_size)
        # scores 的形状为 [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        # 如果传入了 attention_mask，就把 padding 等无效位置屏蔽掉
        if attention_mask is not None:
            mask = self._build_attention_mask(attention_mask, scores.dtype)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        # 对分数做 softmax，得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 用注意力权重对 V 做加权求和，得到上下文向量
        context = torch.matmul(attention_weights, v)

        # 把多头结果重新拼接回 hidden_size
        # 先把形状从 [batch, heads, seq, head_size] 变成 [batch, seq, heads, head_size]
        context = context.transpose(1, 2).contiguous()
        # 再把 heads 和 head_size 合并回 hidden_size
        context = context.view(batch_size, seq_len, self.hidden_size)

        # 最后通过输出线性层
        return self.o(context)

    def _split_heads(self, x, batch_size, seq_len):
        # 原始形状：[batch_size, seq_len, hidden_size]
        # 目标形状：[batch_size, seq_len, num_heads, head_size]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        # 交换 seq_len 和 num_heads 两个维度
        # 变成 [batch_size, num_heads, seq_len, head_size]
        return x.transpose(1, 2)

    def _build_attention_mask(self, attention_mask, dtype):
        # 如果原始 mask 形状是 [batch_size, seq_len]
        # 扩展成 [batch_size, 1, 1, seq_len]，方便和 scores 广播
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        # 如果原始 mask 形状是 [batch_size, seq_len, seq_len]
        # 扩展成 [batch_size, 1, seq_len, seq_len]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        # 其他维度不支持
        elif attention_mask.dim() != 4:
            raise ValueError("attention_mask must have 2, 3, or 4 dimensions")

        # 转成和 scores 相同的数据类型
        return attention_mask.to(dtype=dtype)


class FeedForward(nn.Module):
    # Transformer 中的前馈神经网络
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        # 第一层线性变换：hidden_size -> intermediate_size
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        # BERT 常用 GELU 激活函数
        self.activation = nn.GELU()
        # 第二层线性变换：intermediate_size -> hidden_size
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.activation(x)
        x = self.dense_out(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    # 单层 Transformer Encoder
    def __init__(
        self, hidden_size=768, num_heads=12, dropout=0.1, intermediate_size=3072
    ):
        super().__init__()

        # 第一部分：多头自注意力
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        # 第二部分：前馈网络
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        # 先做自注意力
        attention_output = self.self_attention(x, attention_mask)
        attention_output = self.attention_dropout(attention_output)

        # 残差连接 + LayerNorm
        x = self.attention_layer_norm(x + attention_output)

        # 再做前馈网络
        feed_forward_output = self.feed_forward(x)

        # 再做一次残差连接 + LayerNorm
        x = self.output_layer_norm(x + feed_forward_output)
        return x


if __name__ == "__main__":
    # 下面是一段简单的测试代码，用来验证前向传播和输出形状是否正确
    batch_size = 2
    seq_len = 4
    hidden_size = 768

    # 随机生成输入张量，形状为 [2, 4, 768]
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 构造 attention mask
    # 第一个样本 4 个位置都有效
    # 第二个样本后两个位置是 padding，需要被屏蔽
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]
    )

    # 创建一个 TransformerLayer 实例
    layer = TransformerLayer(
        hidden_size=hidden_size,
        num_heads=12,
        intermediate_size=3072,
        dropout=0.1,
    )

    # 执行前向传播
    output = layer(x, attention_mask)

    # 打印输入和输出形状
    print("input shape:", x.shape)
    print("output shape:", output.shape)
