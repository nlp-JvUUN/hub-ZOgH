import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
    """Transformer块（编码器）"""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # 多头自注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (seq_len, batch_size, d_model)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (seq_len, batch_size, d_model)
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class SimpleTransformer(nn.Module):
    """Transformer模型"""

    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(SimpleTransformer, self).__init__()

        self.d_model = d_model

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)

        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _create_positional_encoding(self, max_len, d_model):
        """位置编码矩阵"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (max_len, 1, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (seq_len, batch_size)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (seq_len, batch_size, vocab_size)
        """
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:x.size(0)]
        x = self.dropout(x)

        # 通过Transformer层
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 输出层
        return self.fc_out(x)


# 测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数
    vocab_size = 1000
    seq_len = 20
    batch_size = 32

    # 创建模型
    model = SimpleTransformer(vocab_size=vocab_size).to(device)

    # 随机输入
    x = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)

    # 前向传播
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
