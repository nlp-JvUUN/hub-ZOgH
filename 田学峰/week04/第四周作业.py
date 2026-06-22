import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q: (batch_size, num_heads, seq_len_q, d_k)
        K: (batch_size, num_heads, seq_len_k, d_k)
        V: (batch_size, num_heads, seq_len_v, d_k)
        mask: (batch_size, 1, seq_len_q, seq_len_k) or broadcastable
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def split_heads(self, x):
        """
        x: (batch_size, seq_len, d_model)
        -> (batch_size, num_heads, seq_len, d_k)
        """
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        x: (batch_size, num_heads, seq_len, d_k)
        -> (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(K)  # (batch_size, seq_len_k, d_model)
        V = self.W_v(V)  # (batch_size, seq_len_v, d_model)
        
        # 拆分多头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # 缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = self.combine_heads(attn_output)  # (batch_size, seq_len_q, d_model)
        
        # 输出线性变换
        output = self.W_o(output)  # (batch_size, seq_len_q, d_model)
        
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: attention mask
        """
        # 自注意力子层
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: attention mask
        """
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        return x, attn_weights_list


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def test_transformer():
    # 参数设置
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 3
    
    # 创建Transformer编码器
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    
    # 创建位置编码
    pos_encoder = PositionalEncoding(d_model)
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
    
    # 添加位置编码（需要转置）
    x_pos = pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, seq_len, d_model)
    
    # 前向传播
    output, attn_weights = encoder(x_pos)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attn_weights)}")
    print(f"Attention weights shape: {attn_weights[0].shape}")
    
    print("\nTransformer layer test passed successfully!")


if __name__ == "__main__":
    test_transformer()
