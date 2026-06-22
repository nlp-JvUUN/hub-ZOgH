"""
用 PyTorch nn.Module 实现一个标准的 Transformer Encoder Layer。
包含：MultiHeadAttention → Add&Norm → FeedForward → Add&Norm
采用 Pre-LN 结构（LayerNorm 在子层之前），训练更稳定。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== MultiHeadAttention ==========================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力，手动计算 Q/K/V/Attention，不使用 nn.MultiheadAttention。
    与 diy_bert.py 中的 self_attention 逻辑一致，只是用 PyTorch 重写。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads               # 每个头的维度

        self.w_q = nn.Linear(d_model, d_model)        # Q 投影
        self.w_k = nn.Linear(d_model, d_model)        # K 投影
        self.w_v = nn.Linear(d_model, d_model)        # V 投影
        self.w_o = nn.Linear(d_model, d_model)        # 输出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x:    (B, T, d_model)
        mask: (T, T) 或 (B, T, T)，True 的位置会被屏蔽（置 -inf）
        返回: (B, T, d_model)
        """
        B, T, _ = x.shape

        # 1. 线性投影
        q = self.w_q(x)   # (B, T, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. 拆分为多头: (B, T, d_model) → (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)

        # 4. 可选 mask（比如 padding mask 或 causal mask）
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # 5. softmax + dropout + 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)        # (B, H, T, d_k)

        # 6. 合并多头: (B, H, T, d_k) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 7. 输出投影 + dropout
        output = self.w_o(context)
        output = self.dropout(output)
        return output


# ========================== FeedForward ==========================

class FeedForward(nn.Module):
    """
    两层全连接前馈网络: d_model → d_ff → d_model
    中间使用 GELU 激活（与 BERT 一致），每层后跟 Dropout。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)                # BERT 使用 GELU，非 ReLU
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ========================== TransformerEncoderLayer ==========================

class TransformerEncoderLayer(nn.Module):
    """
    一个标准的 Transformer 编码器层。
    结构: x → LN → Attention → + → LN → FFN → +   (Pre-LN)
    与 BERT 单层的计算顺序完全一致。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x:    (B, T, d_model)
        mask: 可选，传递给 MultiHeadAttention
        返回: (B, T, d_model)
        """
        # 子层 1: Self-Attention + 残差连接
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # 子层 2: Feed-Forward + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


# ========================== 测试 ==========================

if __name__ == "__main__":
    print("=" * 56)
    print("  Test: PyTorch Transformer Encoder Layer")
    print("=" * 56)

    # 超参（与 bert-base 一致）
    d_model   = 768
    num_heads = 12
    d_ff      = 3072
    dropout   = 0.1
    batch_size = 2
    seq_len   = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    print(f"d_model={d_model}  num_heads={num_heads}  d_ff={d_ff}")

    # 构造模型
    layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout).to(device)

    # 参数量
    total_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}  (trainable: {trainable_params:,})")
    print(f"  Attention params: {sum(p.numel() for n, p in layer.named_parameters() if 'attention' in n):,}")
    print(f"  FFN params:       {sum(p.numel() for n, p in layer.named_parameters() if 'ffn' in n):,}")

    # 打印模型结构摘要
    print(f"\n{'Submodule':<25} {'Class':<25} {'Params':>10}")
    print("-" * 60)
    for name, child in layer.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{name:<25} {child.__class__.__name__:<25} {params:>10,}")

    # 测试 1: 前向传播
    print(f"\n{'=' * 56}")
    print("Test 1: Forward pass")
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"Input shape:   {tuple(x.shape)}")

    layer.eval()
    with torch.no_grad():
        out = layer(x)
    print(f"Output shape:  {tuple(out.shape)}")
    assert tuple(out.shape) == (batch_size, seq_len, d_model), \
        f"Shape mismatch! Expected {(batch_size, seq_len, d_model)}, got {tuple(out.shape)}"
    print("[PASS] Input/output shape match")

    # 测试 2: 反向传播
    print(f"\n{'=' * 56}")
    print("Test 2: Backward pass")
    layer.train()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    grad_norms = {}
    for name, p in layer.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    has_nan = any(math.isnan(v) for v in grad_norms.values())
    has_zero_grads = all(v == 0 for v in grad_norms.values())
    print(f"Gradient params: {len(grad_norms)}")
    print(f"  Mean grad norm: {sum(grad_norms.values()) / len(grad_norms):.6f}")
    print(f"  Max grad norm:  {max(grad_norms.values()):.6f}")
    print(f"  Min grad norm:  {min(grad_norms.values()):.6f}")
    assert not has_nan, "NaN gradients detected!"
    assert not has_zero_grads, "All gradients are zero!"
    print("[PASS] Backward pass OK, no NaN, gradients non-zero")

    # 测试 3: 带 mask 的前向传播
    print(f"\n{'=' * 56}")
    print("Test 3: Forward with causal mask")
    # 上三角 mask，防止看到未来信息（用于 decoder / GPT）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()  # (T, T)
    causal_mask_4d = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    layer.eval()
    with torch.no_grad():
        out_masked = layer(x, mask=causal_mask_4d)
    print(f"Input shape:   {tuple(x.shape)}")
    print(f"Mask shape:    {tuple(causal_mask_4d.shape)}")
    print(f"Output shape:  {tuple(out_masked.shape)}")
    assert tuple(out_masked.shape) == (batch_size, seq_len, d_model)
    print("[PASS] Masked forward pass OK")

    print(f"\n{'=' * 56}")
    print("[ALL PASSED]")
    print(f"{'=' * 56}")
