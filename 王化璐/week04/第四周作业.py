import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 缩放点积注意力（核心基础） --------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: 形状 [batch_size, n_heads, seq_len, d_k]
        mask: 注意力掩码，形状 [batch_size, 1, seq_len, seq_len]
        """
        # 计算注意力分数：Q·K^T / √d_k
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 掩码操作：mask=0的位置置为极小值，softmax后权重为0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重 + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和得到注意力输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# -------------------------- 2. 多头自注意力 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # 约束：模型维度必须能被头数整除
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model  # 模型总维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 单个头的维度

        # 线性层：将输入投影为 Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影层 + 基础注意力
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 步骤1：线性投影 + 拆分多头
        # 形状变换：[batch, seq_len, d_model] → [batch, n_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 步骤2：计算缩放点积注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # 步骤3：拼接多头 + 输出投影
        # 形状变换：[batch, n_heads, seq_len, d_k] → [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dropout(self.w_o(attn_output))

        return output, attn_weights

# -------------------------- 3. 前馈神经网络 --------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 两层线性层：升维 → 激活 → 降维
        self.fc1 = nn.Linear(d_model, d_ff)  # 升维
        self.fc2 = nn.Linear(d_ff, d_model)  # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))  # GELU 比 ReLU 效果更好

# -------------------------- 4. Transformer Encoder 层（最终封装） --------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 核心组件
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # 层归一化（Pre-LN结构）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 残差分支的Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: 输入张量 [batch_size, seq_len, d_model]
        mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]
        返回: 输出张量 [batch_size, seq_len, d_model]
        """
        # ================== 子层1：多头自注意力 + 残差连接 ==================
        attn_output, _ = self.self_attn(x, x, x, mask)  # 自注意力：Q=K=V
        x = self.norm1(x + self.dropout1(attn_output))   # 残差 + 归一化

        # ================== 子层2：前馈网络 + 残差连接 ==================
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))    # 残差 + 归一化

        return x

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 超参数
    d_model = 512    # 模型维度
    n_heads = 8      # 注意力头数
    d_ff = 2048      # 前馈网络中间维度
    batch_size = 2   # 批次大小
    seq_len = 10     # 序列长度

    # 初始化Transformer Encoder层
    encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

    # 构造随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = encoder_layer(x)

    # 打印结果（Transformer层输入输出维度严格一致）
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Transformer Encoder 层实现成功！")
