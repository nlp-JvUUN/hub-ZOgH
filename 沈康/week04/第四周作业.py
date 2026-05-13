"""
第四周作业：
尝试使用pytorch实现一份transformer层
"""
import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 的线性投影层和最终的输出投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放因子，防止点积过大导致梯度消失
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value):
        seq_len = query.size(0)

        # 1. 线性变换并拆分成多头
        Q = self.W_q(query).reshape(seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = self.W_k(key).reshape(seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = self.W_v(value).reshape(seq_len, self.num_heads, self.d_k).swapaxes(1, 2)

        # 2. 计算缩放点积注意力
        scores = torch.matmul(Q, K.swapaxes(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)

        # 3. 加权求和并合并多头
        context = torch.matmul(attn, V)
        context = context.swapaxes(1, 2).reshape(seq_len, -1)

        # 4. 最终线性输出
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    """
    前馈网络
    """

    def __init__(self, d_model):
        super().__init__()
        # 通常中间隐藏层维度是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 线性变换 -> ReLU 激活 -> 线性变换
        return self.linear2(self.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """
    完整的 Transformer 编码器层
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        # 归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 多头自注意力
        attn_output = self.self_attn(x, x, x)
        x = self.layer_norm(x + attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output)
        return x


def main():
    # 假设参数：模型维度512，8个注意力头，序列长度10
    d_model = 512
    num_heads = 8
    seq_len = 10

    # 实例化一个编码器层
    encoder_layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads)

    # 随机生成一个输入张量
    input_tensor = torch.randn(seq_len, d_model)

    # 前向传播
    output_tensor = encoder_layer(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")


if __name__ == "__main__":
    main()
