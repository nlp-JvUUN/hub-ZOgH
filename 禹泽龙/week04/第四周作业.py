"""
Transformer 层实现
=====================================

一步步实现 Transformer 的核心组件：
1. Scaled Dot-Product Attention（缩放点积注意力）
2. Multi-Head Attention（多头注意力）
3. Positional Encoding（位置编码）
4. Transformer Encoder Layer（Transformer编码器层）

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 第一步：Scaled Dot-Product Attention（缩放点积注意力）
# ═══════════════════════════════════════════════════════════════
#
# 原理：
#   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
#
# 为什么需要缩放（除以√d_k）？
#   当d_k较大时，点积的方差会变大，导致softmax后趋于one-hot（梯度消失）
#   缩放因子√d_k使方差回归到1
#
# 直观理解：
#   Q（查询）像是"我正在找什么"
#   K（键）像是"我包含什么信息"
#   V（值）像是"实际要获取的内容"
#   QK^T 计算Q和K的相似度，softmax后得到加权权重，乘以V得到输出

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    参数:
        Q: 查询张量，形状 (batch, seq_len, d_k)
        K: 键张量，形状 (batch, seq_len, d_k)
        V: 值张量，形状 (batch, seq_len, d_v)
        mask: 可选的掩码张量，用于掩盖某些位置

    返回:
        output: 注意力加权后的输出
        attention_weights: 注意力权重矩阵
    """
    d_k = Q.size(-1)  # 获取维度用于缩放

    # 1. 计算Q和K的点积
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. 缩放
    scores = scores / math.sqrt(d_k)

    # 3. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 4. softmax得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 5. 加权求和得到输出
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class SimpleAttention(nn.Module):
    """封装好的注意力模块，用于测试"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, Q, K, V, mask=None):
        return scaled_dot_product_attention(Q, K, V, mask)


# ═══════════════════════════════════════════════════════════════
# 第二步：Multi-Head Attention（多头注意力）
# ═══════════════════════════════════════════════════════════════
#
# 原理：
#   不是做一次注意力，而是做h次（多头），每次有不同的Q,K,V投影
#   多头让模型能同时关注不同类型的信息（如语法、语义、位置等）
#
# 公式：
#   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
#   where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
#
# 直观理解：
#   比如翻译"I love you"到"我爱你"
#   - 一个头可能关注"love"->"爱"的语义关系
#   - 另一个头可能关注"love"和"you"的语法关系
#   - 再一个头可能关注时态或语气

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    参数:
        d_model: 模型维度（必须是num_heads的整数倍）
        num_heads: 注意力头数量
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 假设：batch=2, seq_len=10, d_model=64, num_heads=4, d_k=16
        # 1. 线性投影后分头
        #Q = self.W_Q(Q)                    # (2, 10, 64)
        #Q = Q.view(batch_size, -1, num_heads, d_k)  # (2, 10, 4, 16)
        #Q = Q.transpose(1, 2)              # (2, 4, 10, 16)  头放到维度1
        # 1. 线性投影并分头
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 注意力计算
        #output, _ = scaled_dot_product_attention(Q, K, V)  # (2, 4, 10, 16)
        # 2. 计算注意力
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头
        #output = output.transpose(1, 2)    # (2, 10, 4, 16)
        #output = output.contiguous().view(batch_size, -1, d_model)  # (2, 10, 64)
        # 3. 合并多头：(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. 最终投影
        #output = self.W_O(output)         # (2, 10, 64)
        # 4. 最终线性投影
        output = self.W_O(output)

        return output, attention_weights


# ═══════════════════════════════════════════════════════════════
# 第三步：Positional Encoding（位置编码）
# ═══════════════════════════════════════════════════════════════
#
# 问题：Self-Attention本身是位置无关的，"我爱你"和"你爱我"经过Attention结果相同
# 解决：用位置编码给每个位置添加独特的位置信息
#
# 原理：使用正弦和余弦函数的不同频率
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# 为什么用正弦/余弦？
#   1. 可以表示任意长度的位置（不依赖训练）
#   2. 相对位置可以通过线性变换表示（如PE(pos+k)可以表示为PE(pos)的线性函数）
#   3. 不同频率捕获不同尺度的位置关系

class PositionalEncoding(nn.Module):
    """
    位置编码

    参数:
        d_model: 模型维度
        max_len: 最大序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 计算除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置用sin，奇数位置用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 添加batch维度：(1, max_len, d_model) 便于广播
        pe = pe.unsqueeze(0)

        # 注册为buffer（不参与训练但会保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: 加上位置编码的输出
        """
        return x + self.pe[:, :x.size(1), :]


# 可视化位置编码（调试用）
def plot_positional_encoding():
    """可视化位置编码矩阵"""
    d_model = 64
    max_len = 100

    pe = PositionalEncoding(d_model, max_len)
    pe_matrix = pe.pe[0].numpy()  # (max_len, d_model)

    plt.figure(figsize=(14, 8))
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding (d_model=64, max_len=100)')
    plt.colorbar()
    plt.savefig('positional_encoding.png', dpi=150)
    print("位置编码可视化已保存到 positional_encoding.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 第四步：Feed-Forward Network（前馈网络）
# ═══════════════════════════════════════════════════════════════
#
# 位置前馈网络：每个位置独立经过相同的双层线性变换
# 公式：FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
#
# 为什么需要FFN？
#   Attention主要是线性操作，FFN增加了非线性能力
#   让模型可以学习更复杂的特征变换

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络

    参数:
        d_model: 模型维度
        d_ff: 前馈网络中间层维度（通常4倍d_model）
        dropout: Dropout比例
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ═══════════════════════════════════════════════════════════════
# 第五步：Transformer Encoder Layer（完整的Transformer编码器层）
# ═══════════════════════════════════════════════════════════════
#
# 组成：
#   1. Multi-Head Self-Attention（多头自注意力）
#   2. Add & Layer Norm（残差连接 + 层归一化）
#   3. Feed-Forward Network（前馈网络）
#   4. Add & Layer Norm（残差连接 + 层归一化）
#
# 残差连接（Skip Connection）：
#   输出 = Sublayer(x) + x
#   作用：缓解深层网络的梯度消失问题，让信息直接流动
#
# 层归一化（Layer Normalization）：
#   对每个样本的所有特征做归一化
#   作用：稳定训练、加速收敛

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层

    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络维度
        dropout: Dropout比例
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 两个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # === 第一个子层：Multi-Head Self-Attention ===
        # 残差连接：Attention(x) + x
        attn_output, _ = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # === 第二个子层：Feed-Forward ===
        # 残差连接：FFN(x) + x
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer编码器（多层堆叠）

    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        num_layers: 编码器层数
        d_ff: 前馈网络维度
        dropout: Dropout比例
        vocab_size: 词表大小（用于嵌入层）
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff=2048,
                 dropout=0.1, max_len=5000):
        super().__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 多层编码器
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.dropout(self.pos_encoding(self.embedding(x)))

        # 通过每一层
        for layer in self.layers:
            x = layer(x, mask)

        return x


# ═══════════════════════════════════════════════════════════════
# 演示：用一个小例子测试
# ═══════════════════════════════════════════════════════════════

def demo_attention():
    """演示注意力机制的工作原理"""
    print("\n" + "="*60)
    print("演示1：Scaled Dot-Product Attention")
    print("="*60)

    # 创建一个简单的例子
    batch_size = 1
    seq_len = 4
    d_k = 8

    # 模拟Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"V形状: {V.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"\n注意力权重矩阵（每行表示一个查询对所有键的注意力）:\n{weights[0]}")
    print("(值越大表示越关注)")

    print("\n" + "="*60)
    print("演示2：多头注意力")
    print("="*60)

    d_model = 16
    num_heads = 4
    seq_len = 4
    batch_size = 1

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = mha(x, x, x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")


def demo_transformer():
    """演示完整的Transformer编码器"""
    print("\n" + "="*60)
    print("演示3：完整的Transformer编码器")
    print("="*60)

    # 超参数
    vocab_size = 1000   # 假设词表大小1000
    d_model = 64        # 模型维度
    num_heads = 4       # 4头注意力
    num_layers = 2      # 2层
    d_ff = 128          # 前馈维度（为了演示设小一点）
    seq_len = 16        # 序列长度

    # 创建模型
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )

    # 模拟输入（batch_size=2, seq_len=16）
    batch_size = 2
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    output = model(x)

    print(f"输入形状: {x.shape}  (batch_size, seq_len)")
    print(f"输出形状: {output.shape}  (batch_size, seq_len, d_model)")
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")



if __name__ == "__main__":
    print("="*60)
    print("Transformer 实现")
    print("="*60)

    demo_attention()
    demo_transformer()

    # 可选：生成位置编码可视化
    # plot_positional_encoding()
    print("="*60)
