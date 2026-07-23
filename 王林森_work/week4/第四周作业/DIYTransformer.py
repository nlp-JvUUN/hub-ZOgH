import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#多头自注意，pytorch实现transformer
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("未被整除，具体看dim,heads_num")
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        # 每个头的维度
        self.head_dim = embed_dim // heads_num
        # Q、K、V 的线性变换层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # 输出投影层，将多头拼接后的结果映射回原始维度
        self.output_dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        """
        前向传播流程：
        输入 -> 自注意力 -> 残差+归一化 -> 前馈网络 -> 残差+归一化 -> 输出
        """
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        # 1. 分别计算 Query、Key、Value。
        # 三者形状都为: (batch_size, seq_len, embed_dim)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # 2. 将每个 embedding 拆成多个头。
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 3. 计算注意力得分并缩放
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 4. 对最后一个维度做 softmax，得到每个 token 对其他 token 的注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # 5. 用注意力权重对 Value 加权求和
        context = torch.matmul(attention_weights, value)
        # 6. 把多个头拼接回一个完整 embedding
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.output_dense(context), attention_weights

#Transformer
class TransformerEncoderLayerAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力层
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # 前馈网络
        self.feed_forweard = FeedForward(embed_dim, ff_hidden_dim, dropout)
        # LayerNorm
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    #传播
    def forward(self, x, mask=None):

        # 多头注意操作，
        attention_output, attention_probs = self.attention(x)
        x = self.layer_norm1(x + self.dropout1(attention_output))
        # 前馈操作
        k_output = self.feed_forweard(x)
        x = self.layer_norm2(x + self.dropout2(k_output))

        return x, attention_probs

#前馈网络，包含两个线性层和一个激活函数。
class FeedForward(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # 第一次线性变换
        self.first1 = nn.Linear(embed_dim, hidden_dim)
        # 第二次线性变换
        self.first2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #1线升维度
        up_dimension = self.net1(x)
        #激活
        active = F.relu(up_dimension)
        #2降维
        down_dimension = self.dropout(active)
        x = self.net2(dropoutx)
        return x

def main():
    # 种子一般都是42
    torch.manual_seed(42)
    # 批次大小
    batch_size = 2
    # 序列长度
    seq_len = 6
    # 嵌入维度
    embed_dim = 768
    # 头数
    num_heads = 12
    # 隐藏维度 4倍嵌入维度
    ff_hidden_dim = 3072
    # 构造一个输入例子: 2 条样本，每条样本有 6 个 token，每个 token 用 768 维向量表示。
    x = torch.randn(batch_size, seq_len, embed_dim)
    block = TransformerEncoderLayerAttention(embed_dim=embed_dim,
                             num_heads=num_heads,
                             ff_hidden_dim=ff_hidden_dim,
                             dropout=0.0)
    output, attention_probs = block(x)
    row_sums = attention_probs[0, 0].sum(dim=-1)
    print("注意力权重形状:", attention_probs.shape)
    print("含义: (batch_size, num_heads, seq_len, seq_len)")
    print("注意力权重每行求和（应接近1.0）:", row_sums.tolist())
    print("【结构验证】")
    print(f"模型维度 embed_dim = {embed_dim}")
    print(f"注意力头数 num_heads = {num_heads}")
    print(f"每头维度 d_k = {embed_dim // num_heads}")
    print(f"前馈中间维度 d_ff = {ff_hidden_dim}")
    print(f"参数总量: {sum(p.numel() for p in block.parameters())}")
    print("输入 x 的形状:", x.shape)
    print("输出 output 的形状:", output.shape)


if __name__ == "__main__":
    main()
