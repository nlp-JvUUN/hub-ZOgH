import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size

        # ===== Self-Attention =====
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        # ===== LayerNorm =====
        self.attn_ln = nn.LayerNorm(hidden_size)

        # ===== Feed Forward =====
        self.intermediate = nn.Linear(hidden_size, 3072)
        self.output = nn.Linear(3072, hidden_size)
        self.ffn_ln = nn.LayerNorm(hidden_size)

    # ===== 多头拆分=====
    def transpose_for_scores(self, x):
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    # ===== Self-Attention（完全对齐 DiyBert）=====
    def self_attention(self, x):
        q = self.transpose_for_scores(self.q_linear(x))
        k = self.transpose_for_scores(self.k_linear(x))
        v = self.transpose_for_scores(self.v_linear(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(bsz, seq_len, self.hidden_size)

        return self.attn_out(context)

    # ===== Forward（残差 + LN）=====
    def forward(self, x):
        # Self-Attention
        attn_out = self.self_attention(x)
        x = self.attn_ln(x + attn_out)

        # Feed Forward
        ffn_out = self.output(F.gelu(self.intermediate(x)))
        x = self.ffn_ln(x + ffn_out)

        return x
if __name__ == "__main__":
    bsz, seq_len, hidden_size = 1, 4, 768

    x = torch.randn(bsz, seq_len, hidden_size)

    layer = TransformerEncoderLayer()
    y = layer(x)

    print("Output shape:", y.shape)

