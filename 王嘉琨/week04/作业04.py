import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: 查询向量 [batch_size, n_heads, seq_len_q, d_k]
            k: 键向量 [batch_size, n_heads, seq_len_k, d_k]
            v: 值向量 [batch_size, n_heads, seq_len_v, d_v]
            mask: 掩码 [batch_size, 1, seq_len_q, seq_len_k]（可选）
        
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len_q, d_v]
            attn_weights: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)  # 每个头的维度
        
        # 计算注意力分数：(Q*K^T)/sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, n_heads, seq_len_q, seq_len_k]
        
        # 应用掩码（如填充掩码或序列掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码位置设为极小值
        
        # 计算注意力权重（softmax归一化）
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, seq_len_q, seq_len_k]
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len_q, d_v]
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model  # 输入/输出维度
        self.n_heads = n_heads  # 注意力头数
        
        # 每个头的维度（必须能被d_model整除）
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性变换层（Q、K、V共享权重矩阵）
        self.w_q = nn.Linear(d_model, d_model)  # Q的线性变换
        self.w_k = nn.Linear(d_model, d_model)  # K的线性变换
        self.w_v = nn.Linear(d_model, d_model)  # V的线性变换
        
        # 输出线性变换
        self.w_o = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: 查询 [batch_size, seq_len_q, d_model]
            k: 键 [batch_size, seq_len_k, d_model]
            v: 值 [batch_size, seq_len_v, d_model]
            mask: 掩码 [batch_size, seq_len_q, seq_len_k]（可选）
        
        Returns:
            output: 多头注意力输出 [batch_size, seq_len_q, d_model]
            attn_weights: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        
        # 线性变换 + 分拆多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len_q, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len_k, d_k]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # [batch_size, n_heads, seq_len_v, d_v]
        
        # 调整掩码维度以适应多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
        
        # 计算多头注意力
        output, attn_weights = self.attention(q, k, v, mask)  # [batch_size, n_heads, seq_len_q, d_v]
        
        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
        
        # 输出线性变换
        output = self.w_o(output)  # [batch_size, seq_len_q, d_model]
        
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换
        self.dropout = nn.Dropout(dropout)   # Dropout层
        self.activation = nn.GELU()         # 激活函数（原论文用ReLU，这里用更优的GELU）

    def forward(self, x):
        """
        Args:
            x: 输入 [batch_size, seq_len, d_model]
        
        Returns:
            输出 [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)       # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)       # [batch_size, seq_len, d_model]
        return x


class TransformerLayer(nn.Module):
    """完整的Transformer层（编码器层/解码器层基础）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
            mask: 自注意力掩码 [batch_size, seq_len, seq_len]（可选）
        
        Returns:
            output: Transformer层输出 [batch_size, seq_len, d_model]
            attn_weights: 自注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attn(x, x, x, mask)  # Q=K=V（自注意力）
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))    # 残差连接
        
        return x, attn_weights


# 测试代码
if __name__ == "__main__":
    # 超参数
    d_model = 512    # 模型维度
    n_heads = 8      # 注意力头数
    d_ff = 2048      # 前馈网络隐藏层维度
    seq_len = 10     # 序列长度
    batch_size = 2   # 批次大小
    
    # 初始化Transformer层
    transformer_layer = TransformerLayer(d_model, n_heads, d_ff)
    
    # 随机生成输入数据
    x = torch.randn(batch_size, seq_len, d_model)  # [2, 10, 512]
    
    # 生成掩码（示例：假设前5个位置有效，后5个为填充）
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, 5:] = 0  # 掩盖序列后半部分
    
    # 前向传播
    output, attn_weights = transformer_layer(x, mask)
    
    # 输出形状检查
    print(f"输入形状: {x.shape}")                # [2, 10, 512]
    print(f"输出形状: {output.shape}")            # [2, 10, 512]
    print(f"注意力权重形状: {attn_weights.shape}")  # [2, 8, 10, 10]
