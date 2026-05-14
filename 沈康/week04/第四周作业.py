"""
第四周作业：
尝试使用pytorch实现一份transformer层
"""
import math

import torch
import torch.nn as nn


# 多头自注意力机制 (Multi-Head Self-Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 的线性投影层和最终的输出投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放因子，防止点积过大导致梯度消失
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性变换并拆分成多头
        Q = self.W_q(query).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = self.W_k(key).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = self.W_v(value).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.swapaxes(-2, -1)) / self.scale

        # 应用掩码 (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        attn = torch.softmax(scores, dim=-1)

        # 加权求和并合并多头
        context = torch.matmul(attn, V)
        context = context.swapaxes(1, 2).reshape(batch_size, seq_len, -1)

        # 最终线性输出
        return self.W_o(context)


# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 通常中间隐藏层维度是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 线性变换 -> ReLU 激活 -> 线性变换
        return self.linear2(self.relu(self.linear1(x)))


# 位置编码 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# Transformer 编码器层 (Encoder Layer)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)

        # 归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 多头自注意力
        attn_output = self.self_attn(x, x, x)
        x = self.layer_norm(x + attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output)
        return x


# Transformer 解码器层 (Decoder Layer)
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)

        # 归一化
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 带掩码的自注意力 (Masked Self-Attention)
        x = x + self.dropout(self.self_attn(self.layer_norm(x), x, x, tgt_mask))

        # 交叉注意力 (Cross-Attention)
        x = x + self.dropout(self.cross_attn(self.layer_norm(x), enc_output, enc_output, src_mask))

        # 前馈网络
        x = self.layer_norm(x + self.dropout(self.feed_forward(x)))
        return x


# 完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠 N 个 Encoder 和 Decoder 层
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 嵌入与位置编码
        src_embed = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_embed = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 编码器前向传播
        enc_output = src_embed
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        # 解码器前向传播
        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # 输出层投影到词表大小
        output = self.fc_out(dec_output)
        return output


def main():
    # 设置参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    src_len = 10
    tgt_len = 10
    batch_size = 2

    # 实例化模型
    model = Transformer(src_vocab_size, tgt_vocab_size)

    # 创建简单的掩码
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(1)

    # 模拟输入数据
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 前向传播
    output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)

    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")


if __name__ == "__main__":
    main()
