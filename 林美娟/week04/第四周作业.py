#尝试利用pytorch实现一个Transformer层
# 包含多头自注意力机制和位置前馈神经网络两个主要子层，每个子层后均跟随残差连接和层归一化操作。

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1):
        """
        初始化 Transformer 层。

        Args:
            hidden_size (int): 隐藏层的维度大小，默认为 768。
            num_attention_heads (int): 注意力头的数量，默认为 12。
            intermediate_size (int): 前馈网络中间层的维度大小，默认为 3072。
            dropout (float): Dropout 的概率，默认为 0.1。
        """
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # Self-attention层：定义查询、键、值线性变换及注意力输出线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)

        # LayerNorm for attention：注意力机制后的层归一化
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        # Feed-forward层：定义前馈网络的两层线性变换
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

        # LayerNorm for feed-forward：前馈网络后的层归一化
        self.ff_layer_norm = nn.LayerNorm(hidden_size)

        # Dropout：：定义丢弃层用于正则化
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        将输入张量重塑并转置，以便进行多头注意力计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size]

        返回:
            torch.Tensor: 转置后的张量，形状为 [batch_size, num_heads, seq_len, head_size]
        """
        # x.shape = [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.shape
        # 将隐藏层维度拆分为多头和每个头的维度
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        # 调整维度顺序，使头维度位于序列长度之前，便于后续注意力分数计算
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]
        return x

    def forward(self, x):
        """
        执行Transformer编码器层的前向传播。

        该过程包含多头自注意力机制和位置前馈神经网络两个主要子层，
        每个子层后均跟随残差连接和层归一化操作。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size]。

        Returns:
            torch.Tensor: 经过编码层处理后的输出张量，形状为 [batch_size, seq_len, hidden_size]。
        """
        # x.shape = [batch_size, seq_len, hidden_size]

        # Self-attention
        q = self.query(x)  # [batch_size, seq_len, hidden_size]
        k = self.key(x)    # [batch_size, seq_len, hidden_size]
        v = self.value(x)  # [batch_size, seq_len, hidden_size]

        # Multi-head attention
        q = self.transpose_for_scores(q)  # [batch_size, num_heads, seq_len, head_size]
        k = self.transpose_for_scores(k)  # [batch_size, num_heads, seq_len, head_size]
        v = self.transpose_for_scores(v)  # [batch_size, num_heads, seq_len, head_size]

        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / (self.attention_head_size ** 0.5) #/ sqrt(head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context layer：加权 V → 换维度顺序 → 把多头拼回完整维度
        context_layer = torch.matmul(attention_probs, v)  # [batch_size, num_heads, seq_len, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_size]
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)  # [batch_size, seq_len, hidden_size]

        # Attention output
        attention_output = self.attention_output(context_layer)
        attention_output = self.dropout(attention_output)

        # Residual connection and layer norm
        x = self.attention_layer_norm(x + attention_output)

        # Feed-forward
        intermediate_output = self.intermediate(x)
        intermediate_output = F.gelu(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)

        ff_output = self.output(intermediate_output)
        ff_output = self.dropout(ff_output)

        # Residual connection and layer norm
        x = self.ff_layer_norm(x + ff_output)

        return x

# 测试一层Transformer
if __name__ == "__main__":
    # 创建模型
    transformer_layer = TransformerLayer()

    # 假设输入是embedding后的结果，shape=[batch_size, seq_len, hidden_size]
    batch_size = 1
    seq_len = 4
    hidden_size = 768
    x_input = torch.randn(batch_size, seq_len, hidden_size)

    # 前向传播
    output = transformer_layer(x_input)
    print(f"Input shape: {x_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer layer test passed!")
