# ============================================================
# 1. 导入
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 2. 模拟文本数据 + 字符级分词
# ============================================================
prompt = "很多公司开始训练自己的"
text = """
人工智能正在快速改变世界。近年来，大模型技术取得了巨大的突破，
自然语言处理、计算机视觉、语音识别等领域都出现了明显进展。

很多公司开始训练自己的语言模型，希望能够构建智能助手、
搜索系统、自动写作工具以及代码生成系统。

Transformer模型已经成为现代人工智能系统的重要基础。
它最核心的结构是自注意力机制。模型能够在处理一个词的时候，
同时关注句子中的其他词，从而理解上下文关系。

GPT是一种单向的Transformer模型。
它采用了因果掩码，也叫三角形掩码。
模型在预测下一个token时，只允许看到当前位置之前的内容，
而不能看到未来的信息。

例如，当模型读到“今天天气很好”时，
如果要预测下一个字，它只能看到“今天天气很好”，
不能提前偷看后面的内容。

语言模型的训练目标通常非常简单。
输入一句话，模型负责预测下一个token。
训练过程中会不断计算预测结果与真实结果之间的误差，
然后使用反向传播更新参数。

深度学习模型通常需要大量数据。
新闻、小说、网页、论文、代码都可以作为训练语料。
数据越丰富，模型通常越容易学到复杂的语言规律。

中文训练和英文训练有一些不同。
英文天然以空格分词，而中文没有空格，
因此通常需要额外的tokenizer。

最简单的方法是字符级建模。
也就是把每一个汉字当作一个token。
虽然效率不高，但是逻辑最简单，
非常适合理解GPT的工作原理。

在生成文本时，模型会先读取一个prompt。
随后不断预测下一个token，
再把预测结果拼接回输入，
继续预测后面的token。

这个过程称为自回归生成。

如果模型规模足够大，训练数据足够多，
它甚至能够表现出推理、总结、翻译、
代码生成以及复杂问答等能力。

目前，大模型已经广泛应用于教育、
医疗、金融、工业、娱乐等多个领域。

未来，人工智能可能会成为像电力和互联网一样的重要基础设施。
越来越多的人开始学习机器学习、
深度学习以及Transformer相关技术。

很多初学者第一次接触GPT时，
都会觉得Transformer结构比较复杂。
但实际上，核心逻辑并不多。

输入token。
查embedding。
经过masked self-attention。
经过前馈网络。
预测下一个token。

不断重复这个过程，
模型就能够逐渐学会语言规律。

因此，理解一个最小版本的GPT，
是学习大模型原理非常重要的一步。
"""

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    return ''.join([itos[int(i)] for i in ids])

data = encode(text)


# ============================================================
# 3. 构造训练 batch
# ============================================================

block_size = 32
batch_size = 16

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ============================================================
# 4. Transformer 模型组件
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = q @ k.transpose(-2, -1)
        att = att / (self.head_dim ** 0.5)

        # 三角形 causal mask：当前位置只能看自己和之前的 token
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================
# 5. GPT 风格语言模型
# ============================================================

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()

        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = token + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ============================================================
# 6. 训练
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1000):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.4f}")


# ============================================================
# 7. 预测 / 生成
# ============================================================


idx = encode(prompt).unsqueeze(0).to(device)

out = model.generate(idx, max_new_tokens=100)[0]
print(decode(out))
