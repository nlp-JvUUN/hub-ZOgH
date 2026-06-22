#洪家明week4作业

'''

（本周第四周作业题目：）
尝试用pytorch实现一个transformer层

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # 线性变换并拆分成多头
        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # 加权求和并合并多头
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model=512, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头注意力 + 残差 + 层归一化
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# 使用示例
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 512
    layer = TransformerLayer()
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)

    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")