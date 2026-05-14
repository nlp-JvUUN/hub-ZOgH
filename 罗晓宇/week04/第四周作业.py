'''
使用pytorch实现一个transformer层：

结构：
Multi-Head Attention + Add & Norm + Feed Forward + Add & Norm

'''

import torch
import torch.nn as nn


# 自行实现的TransformerLayer
class DiyTransformerLayer:
    # 初始化，生成权重矩阵
    def __init__(self, L, d_model, num_heads=12, intermediate_size=3072):
        self.L = L
        self.d_model = d_model
        # 初始化多头注意力的头数和每个头的维度
        self.num_heads = num_heads
        self.d_k = d_model // self.num_heads
        # 初始化FFN中间层的维度
        self.d_ff = intermediate_size

        # 生成权重矩阵
        # 注意力权重矩阵：Q, K, V 的权重矩阵
        self.q_w = nn.Parameter(torch.rand(d_model, d_model))
        self.q_b = nn.Parameter(torch.rand(d_model))
        self.k_w = nn.Parameter(torch.rand(d_model, d_model))
        self.k_b = nn.Parameter(torch.rand(d_model))
        self.v_w = nn.Parameter(torch.rand(d_model, d_model))
        self.v_b = nn.Parameter(torch.rand(d_model))
        self.O_w = nn.Parameter(torch.rand(d_model, d_model))
        self.O_b = nn.Parameter(torch.rand(d_model))
        # 前馈神经网络的权重矩阵
        self.ff1_w = nn.Parameter(torch.rand(d_model, intermediate_size))
        self.ff2_w = nn.Parameter(torch.rand(intermediate_size, d_model))
        self.ff1_b = nn.Parameter(torch.rand(intermediate_size))
        self.ff2_b = nn.Parameter(torch.rand(d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    # Multi-Head Self-Attention
    def multi_head_attention(self, x):
        # x shape: (L, d_model)
        q = x @ self.q_w.t() + self.q_b  # (L, d_model)
        k = x @ self.k_w.t() + self.k_b  # (L, d_model)
        v = x @ self.v_w.t() + self.v_b  # (L, d_model)
        # 将 q, k, v 分成 num_heads 个头，每个头的维度为 d_k
        q = q.view(self.L, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, L, d_k)
        k = k.view(self.L, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, L, d_k)
        v = v.view(self.L, self.num_heads, self.d_k).permute(1, 0, 2)  # (num_heads, L, d_k)
        # 计算注意力分数矩阵
        qk = torch.matmul(q, k.transpose(1, 2)) / (self.d_k ** 0.5)  # (num_heads, L, L)
        qk = nn.functional.softmax(qk, dim=2)  # (num_heads, L, L)
        # 计算加权值矩阵
        qkv = torch.matmul(qk, v)  # (num_heads, L, d_k)
        # 将多个头的输出拼接起来
        qkv = qkv.permute(1, 0, 2).reshape(self.L, self.d_model)  # (L, d_model)
        # 线性变换得到最终的输出
        output = qkv @ self.O_w.t() + self.O_b  # (L, d_model)
        return output
    
    # 前馈神经网络
    def feed_forward(self, x):
        ff1 = x @ self.ff1_w + self.ff1_b  # (L, intermediate_size)
        ff1 = nn.functional.gelu(ff1)  # GELU 激活函数
        ff2 = ff1 @ self.ff2_w + self.ff2_b  # (L, d_model)
        return ff2

    # 前向传播
    def forward(self, x):
        # Multi-Head Attention
        attn_output = self.multi_head_attention(x)  # (L, d_model)
        # Add & Norm
        x = self.norm1(x + attn_output)  # (L, d_model)
        # Feed Forward
        ff_output = self.feed_forward(x)  # (L, d_model)
        # Add & Norm
        output = self.norm2(x + ff_output)  # (L, d_model)
        return output

# 测试代码
if __name__ == "__main__":
    L = 128  # 序列长度
    d_model = 768  # 模型维度
    intermediate_size = 3072  # FFN中间层维度
    transformer_layer = DiyTransformerLayer(L, d_model, intermediate_size=intermediate_size)
    x = torch.rand(L, d_model)  # 输入数据
    output = transformer_layer.forward(x)
    print("输出形状:", output.shape)  # 应该是 (L, d_model)
