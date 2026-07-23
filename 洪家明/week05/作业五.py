#洪家明week5作业

'''

【第五周作业】
训练基于transformer的单向语言模型，并完成文本生成

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)   # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # [B, T, 3H] -> 3 个 [B, n_head, T, d_k]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # scaled dot-product
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v                              # [B, n_head, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))        # 残差 + LN
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12, ff=3072):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MiniLM(nn.Module):
    def __init__(self, vocab_size, hidden=512, n_layer=6, n_head=8, ff=1024, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.backbone = TransformerEncoder(hidden, n_layer, n_head, ff)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        # 位置索引 (0..T-1)
        pos = torch.arange(0, T, device=device).unsqueeze(0)           # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)                    # 嵌入 + 位置
        # 因果掩码 (下三角)
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        x = self.backbone(x, mask)                                     # 单向注意力
        logits = self.lm_head(x)                                       # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        device = next(self.parameters()).device
        idx = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            if idx.size(1) > self.max_len:
                idx = idx[:, -self.max_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx.squeeze(0)

if __name__ == "__main__":
    model = MiniLM(vocab_size=100, hidden=128, n_layer=4, n_head=4, ff=256, max_len=64)
    src = torch.randint(0, 100, (2, 16))
    tgt = torch.randint(0, 100, (2, 16))
    logits, loss = model(src, tgt)
    print(f"loss: {loss.item():.4f}")

    prompt = [5, 12, 8]
    generated = model.generate(prompt, 20, temperature=0.8, top_k=5)
    print(f"generated tokens: {generated.tolist()}")
