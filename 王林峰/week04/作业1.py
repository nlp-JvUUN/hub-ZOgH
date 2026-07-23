import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 1. 缩放点积注意力（基础单元）
# --------------------------
def scaled_dot_product_attention(q: torch.Tensor, 
                                 k: torch.Tensor, 
                                 v: torch.Tensor, 
                                 mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算缩放点积注意力
    Args:
        q: 查询向量 [batch_size, n_heads, seq_len, d_k]
        k: 键向量 [batch_size, n_heads, seq_len, d_k]
        v: 值向量 [batch_size, n_heads, seq_len, d_v]
        mask: 掩码张量 [batch_size, 1, seq_len, seq_len]
    Returns:
        注意力输出 + 注意力权重
    """
    d_k = q.size(-1)
    # 1. 计算 Q*K^T / sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 2. 掩码（可选，防止看到未来token）
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    
    # 3. Softmax 得到注意力权重
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # 4. 加权求和得到输出
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

# --------------------------
# 2. 多头自注意力
# --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model    # 模型总维度
        self.n_heads = n_heads    # 注意力头数
        assert d_model % n_heads == 0, "模型维度必须能被头数整除"
        
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 3个线性层：生成 Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        拆分多头：[batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        """
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性变换生成 Q/K/V
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # 2. 拆分成多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # 3. 计算缩放点积注意力
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 4. 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终投影
        output = self.w_o(attn_output)
        return output, attn_weights

# --------------------------
# 3. 前馈网络 FFN
# --------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 升维
        self.fc2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# --------------------------
# 4. 完整 Transformer 编码器层
# --------------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 多头注意力
        self.mha = MultiHeadAttention(d_model, n_heads)
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 两个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        标准 Transformer 层前向传播
        结构：LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
        """
        # 1. 自注意力 + 残差 + 归一化
        attn_output, _ = self.mha(x, x, x, mask)  # 自注意力：Q=K=V
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差 + 归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
    
# 超参数
d_model = 512    # 模型维度
n_heads = 8      # 注意力头数
d_ff = 2048      # 前馈网络中间维度
batch_size = 2   # 批次大小
seq_len = 10     # 序列长度

# 初始化 Transformer 层
transformer_layer = TransformerLayer(d_model, n_heads, d_ff)

# 生成随机输入 [batch, seq_len, d_model]
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output = transformer_layer(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # 输出形状和输入完全一致