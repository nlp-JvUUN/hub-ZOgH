
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # 因果掩码：下三角，防止看到未来信息
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = MultiHeadCausalAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CausalLM(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layer=12, n_head=12, ff=3072, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pos_enc = PositionalEncoding(hidden, max_len)
        self.layers = nn.ModuleList([DecoderLayer(hidden, n_head, ff) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.hidden = hidden

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids) * math.sqrt(self.hidden)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


if __name__ == "__main__":
    # 简单字符级语言模型示例
    vocab_size = 100
    model = CausalLM(vocab_size=vocab_size, hidden=128, n_layer=4, n_head=4, ff=256)

    # 测试前向传播
    input_ids = torch.randint(0, vocab_size, (2, 16))
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # 测试文本生成
    seed = torch.tensor([[1, 2, 3]])
    generated = model.generate(seed, max_new_tokens=20, temperature=1.0)
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"Generated length: {generated.size(1)}")
