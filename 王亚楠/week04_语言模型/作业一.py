import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, dropout):
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
  def forward(self, x, mask = None):
    norm_x = self.norm1(x)
    mha_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask = mask)
    x = x + mha_out
    x = x + self.ffn(self.norm2(x))
    return x

def test_transformer_block():
    # 1. 设定参数
    batch_size = 2
    seq_len = 10
    embed_dim = 128
    num_heads = 8
    ff_dim = 512
    dropout = 0.1

    # 2. 初始化模型
    model = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
    model.train() # 开启 train 模式以激活 Dropout

    # 3. 构造模拟输入 (Batch, Seq_Len, Embed_Dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"输入形状: {x.shape}")

    # 4. 测试基础前向传播
    try:
        output = model(x)
        print(f"输出形状: {output.shape}")
        assert output.shape == x.shape, "形状不匹配！"
        print("✅ 基础前向传播测试通过！")
    except Exception as e:
        print(f"❌ 基础前向传播失败: {e}")
        return

    # 5. 测试带 Mask 的前向传播 (模拟因果掩码/上三角矩阵)
    # attn_mask 形状通常为 (seq_len, seq_len) 或 (batch*num_heads, seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() # True 表示被掩码掉
    try:
        output_masked = model(x, mask=mask)
        print("✅ 带 Mask 的前向传播测试通过！")
    except Exception as e:
        print(f"❌ Mask 测试失败: {e}")

    # 6. 测试反向传播 (梯度检查)
    try:
        loss = output.mean()
        loss.backward()
        # 检查其中一个参数是否有梯度
        has_grad = model.mha.in_proj_weight.grad is not None
        print(f"是否存在梯度: {has_grad}")
        assert has_grad, "没有计算出梯度！"
        print("✅ 反向传播及梯度计算通过！")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")

if __name__ == "__main__":
    test_transformer_block()
