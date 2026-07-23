import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.haed_dim = d_model // num_heads

        # Q K V projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            B = query.size(0)

            # Linear Projection
            Q = self.q_proj(query)
            K = self.k_proj(key)
            V = self.v_proj(value)

            # Split Heads
            # [B, L, D] -> [B, H, L, head_dim]
            Q = Q.view(B, self.num_heads, -1 , self.head_dim).transpose(1, 2)
            K = K.view(B, self.num_heads, -1 , self.head_dim).transpose(1, 2)
            V = V.view(B, self.num_heads, -1 , self.head_dim).transpose(1, 2)

            # Scaled Dot Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, V)

            # Concatenate Heads
            out = out.transpose(1, 2).contiguous()
            out = out.view(B, -1, self.d_model)

            out = self.out_proj(out)
            return out



class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# Input->multi-head self attention->add & LayerNorm -> Feed forward -> add & LayerNorm -> output
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # self attention
        attn_out = self.self_attn(x, x, x, src_mask)
        
        # add & LayerNorm
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.ffn(x)
        
        # add & LayerNorm
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


# Input->multi-head self attention->add & LayerNorm -> Cross-attention -> add & LayerNorm -> Feed forward -> add & LayerNorm -> output
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # masked self attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # encoder-decoder attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # masked self attention
        attn_out = self.self_attn(x, x, x, tgt_mask)

        # add & LayerNorm
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Cross-attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)

        # add & LayerNorm
        x = x + self.dropout(cross_out)
        x = self.norm2(x)

        # FFN
        ffn_out = self.ffn(x)

        # add & LayerNorm
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)

        return x



def generate_square_subsequent_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)



if __name__ == "__main__":
    batch_size = 2
    src_len = 10
    tgt_len = 8

    d_model = 512
    num_heads = 8
    d_ff = 2048

    # encoder input
    src = torch.randn(batch_size, src_len, d_model)

    # decoder input
    tgt = torch.randn(batch_size, tgt_len, d_model)

    # encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    encoder_output = encoder_layer(src)
    print("Encoder Output shape:")
    print(encoder_output.shape)

    # decoder layer
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

    # causal mask
    tgt_mask = generate_square_subsequent_mask(tgt_len)

    decoder_output = decoder_layer(tgt, encoder_output, tgt_mask=tgt_mask)
    print("Decoder Output shape:")
    print(decoder_output.shape)
