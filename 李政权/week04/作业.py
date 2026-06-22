import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch * num_heads, seq_len, d_k)
        mask:    (batch * num_heads, seq_len, seq_len) or broadcastable
        """
        d_k = Q.size(-1)

        # (batch * heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask 需要是 2D 或 3D，用于屏蔽 attention scores
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len)
        """
        B = x.size(0)

        # (B, seq_len, num_heads, d_k)
        Q = self.W_q(x).view(B, -1, self.num_heads, self.d_k)
        K = self.W_k(x).view(B, -1, self.num_heads, self.d_k)
        V = self.W_v(x).view(B, -1, self.num_heads, self.d_k)

        # (B * num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2).contiguous().view(B * self.num_heads, -1, self.d_k)
        K = K.transpose(1, 2).contiguous().view(B * self.num_heads, -1, self.d_k)
        V = V.transpose(1, 2).contiguous().view(B * self.num_heads, -1, self.d_k)

        # 修复点：mask 需要从 (B, seq_len) -> (B*heads, seq_len, seq_len)
        if mask is not None:
            # (B, 1, 1, seq_len) -> (B, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.repeat(1, self.num_heads, 1, 1)
            mask = mask.view(B * self.num_heads, -1, mask.size(-1))

        attn_output, _ = self.attention(Q, K, V, mask)

        # reshape back
        attn_output = attn_output.view(B, self.num_heads, -1, self.d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, self.num_heads * self.d_k)

        return self.W_o(attn_output)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, input_ids, segment_ids=None):
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        pos_ids = torch.arange(
            0, input_ids.size(1),
            device=input_ids.device
        ).unsqueeze(0)

        embeddings = (
                self.token_embedding(input_ids)
                + self.segment_embedding(segment_ids)
                + self.position_embedding(pos_ids)
        )
        return self.dropout(embeddings)


class BertEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, segment_ids=None, mask=None):
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


def test_bert_encoder():
    vocab_size = 30000
    d_model = 768
    num_heads = 12
    d_ff = 3072
    num_layers = 12
    max_seq_len = 512
    batch_size = 2
    seq_len = 128

    model = BertEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, mask=mask)
    print("Input shape: ", input_ids.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_bert_encoder()
