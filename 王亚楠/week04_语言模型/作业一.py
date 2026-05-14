import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, dropout):
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heas, nn.Dropout(dropout))
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.sequential(nn.Linear(embed_dim, ff_dim), nn.Dropout(droport), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
  def forward(self, x, mask = None):
    mha_out = self.mah(self.norm1(x), mask = mask)
    x = x + mha_out
    x = x + self.ffn(self.norm2(x))
    return x
