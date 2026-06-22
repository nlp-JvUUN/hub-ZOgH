import torch
import torch.nn as nn
import torch.nn.functional as torch_func
import math
'''
torch 实现一个 Transformer 层
Transformer 层由两个子层组成：
1. 输入向量

2. Multi-Head Self-Attention：
    1. 输入序列经过线性映射，得到 Q、K、V 三个矩阵
    2. 将 Q、K 矩阵相乘，得到 注意力分数矩阵
    3. 将 注意力分数矩阵 通过 softmax 激活函数归一化，得到注意力权重
    4. 将 V 矩阵与注意力权重矩阵相乘，得到注意力输出
    5. 将注意力输出与输入序列进行拼接，得到输出序列
3. 残差连接 + 层归一化
    输入序列与残差连接后，经过层归一化，得到输出序列
4. 前馈网络（Feed-Forward Network）：
    多层感知机（MLP）
    输入序列经过线性映射，得到 FFN 输入
    将 FFN 输入通过 GELU 激活函数，得到输出
    将输出与输入序列进行拼接，得到输出序列
5. 残差连接 + 层归一化
    输入序列与残差连接后，经过层归一化，得到输出序列
6. 输出：
    输出序列
Transformer 层的输出与输入序列维度相同，可以与其他层进行拼接。

'''
# ================================
# 1. 多头自注意力层 Multi-Head Self-Attention
# ================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性映射得到 Q, K, V
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        # 输出映射
        self.w_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape  # batch, seq_len, embed_dim

        # ======================
        # 步骤1：线性映射得到 Q、K、V
        # ======================
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 拆分成多头
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ======================
        # 步骤2：Q × K^T 得到注意力分数
        # ======================
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ======================
        # 步骤3：softmax 归一化 → 注意力权重
        # ======================
        attn_weight = torch_func.softmax(attn_score, dim=-1)

        # ======================
        # 步骤4：权重 × V → 注意力输出
        # ======================
        attn_out = torch.matmul(attn_weight, V)

        # 拼接多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # ======================
        # 步骤5：线性映射输出（与输入序列维度一致）
        # ======================
        out = self.w_o(attn_out)
        return out


# ================================
# 2. 前馈网络 FFN（MLP + GELU）
# ================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        # 线性映射
        x = self.linear1(x)
        # GELU 激活
        x = self.gelu(x)
        # 线性映射回原维度
        x = self.linear2(x)
        return x


# ================================
# transformer 层
# ================================
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4

        # 子层 1 Multi-Head Self-Attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        # 残差连接 + 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)

        # 子层 2 Feed Forward Network
        self.ffn = FeedForward(embed_dim, ff_dim)
        # 残差连接 + 层归一化
        self.norm2 = nn.LayerNorm(embed_dim)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 输入向量：x
        # 2. 多头自注意力层 Multi-Head Self-Attention
        attn_out = self.attn(x)
        # 3. 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(attn_out))
        # 4. 前馈网络 Feed Forward
        ffn_out = self.ffn(x)
        # 5. 残差连接 + 层归一化
        x = self.norm2(x + self.dropout(ffn_out))
        # 6. 输出（和输入形状一样）
        return x


# ================================
# 测试运行
# ================================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 8

    # 输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 构建 transformer 层
    transformer = TransformerLayer(embed_dim, num_heads)

    # 前向传播
    out = transformer(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)
