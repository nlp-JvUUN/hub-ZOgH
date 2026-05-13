import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads

        # 线性层：将输入映射到Q、K、V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 输出线性层
        self.output = nn.Linear(hidden_size, hidden_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        将输入转换为多头格式
        输入: [batch_size, seq_len, hidden_size]
        输出: [batch_size, num_heads, seq_len, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len] 或 [batch_size, seq_len]
        Returns:
            attention_output: [batch_size, seq_len, hidden_size]
        """
        # 1. 线性变换得到Q、K、V
        mixed_query_layer = self.query(hidden_states)  # [batch, seq_len, hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [batch, seq_len, hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [batch, seq_len, hidden_size]

        # 2. 转换为多头格式
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch, heads, seq_len, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [batch, heads, seq_len, head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch, heads, seq_len, head_size]

        # 3. 计算注意力分数 Q × K^T / √d_k
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch, heads, seq_len, seq_len]
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        # 4. 应用注意力掩码（可选）
        if attention_mask is not None:
            # 如果掩码是2D的，扩展为4D
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + (attention_mask * -10000.0)

        # 5. Softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch, heads, seq_len, seq_len]

        # 6. Dropout（可选）
        attention_probs = self.dropout(attention_probs)

        # 7. 计算输出 QKV = attention × V
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch, heads, seq_len, head_size]

        # 8. 转换回原始形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        attention_output = context_layer.view(*new_context_layer_shape)  # [batch, seq_len, hidden_size]

        # 9. 输出线性层
        attention_output = self.output(attention_output)

        return attention_output


class FeedForward(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        x = self.dense1(x)  # [batch, seq_len, intermediate_size]
        x = F.gelu(x)  # GELU激活函数
        x = self.dropout(x)  # Dropout
        x = self.dense2(x)  # [batch, seq_len, hidden_size]
        return x


class TransformerLayer(nn.Module):
    """
    Transformer层（Encoder层）
    包含：多头注意力 + 残差连接 + LayerNorm + 前馈网络 + 残差连接 + LayerNorm
    """

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)

        # 注意力输出的LayerNorm
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # 前馈网络
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)

        # 前馈输出的LayerNorm
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 或 None
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # 1. 多头注意力
        attention_output = self.attention(hidden_states, attention_mask)  # [batch, seq_len, hidden_size]

        # 2. 残差连接 + LayerNorm
        attention_output = self.dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)  # [batch, seq_len, hidden_size]

        # 3. 前馈网络
        feed_forward_output = self.feed_forward(hidden_states)  # [batch, seq_len, hidden_size]

        # 4. 残差连接 + LayerNorm
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.feed_forward_norm(hidden_states + feed_forward_output)  # [batch, seq_len, hidden_size]

        return output


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 超参数设置（与BERT-base一致）
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072  # 通常是hidden_size的4倍

    # 创建Transformer层
    transformer_layer = TransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size
    )

    # 创建随机输入
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    # 创建注意力掩码（可选）
    # 假设前5个token是有效token，后面是padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 5:] = 0  # padding部分设为0

    # 前向传播
    output = transformer_layer(input_tensor, attention_mask)

    # 输出结果
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Transformer层参数数量: {sum(p.numel() for p in transformer_layer.parameters())}")
