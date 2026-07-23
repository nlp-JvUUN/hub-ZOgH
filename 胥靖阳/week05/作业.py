# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/5/21 19:22  
@脚本说明 : 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from collections import Counter

# -------------------- 1. 单向多头注意力（带因果掩码）--------------------
class CausalMultiHeadAttention(nn.Module):
    """单向多头自注意力：只能看到当前位置及之前的 token"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # 1. 线性变换并拆分为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 生成因果掩码（下三角矩阵）
        if mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        else:
            causal_mask = mask  # 可以传入外部掩码（与因果掩码结合）

        scores = scores.masked_fill(causal_mask == 0, -1e9)

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # 6. 合并多头并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        return output


# -------------------- 2. 逐位置前馈网络 --------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# -------------------- 3. Transformer 解码器层（单向）--------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = CausalMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN 结构
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = residual + x
        return x


# -------------------- 4. 完整语言模型 --------------------
class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=512, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        # 位置编码
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits

    def generate(self, start_token_id, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成文本"""
        self.eval()
        generated = [start_token_id]
        device = next(self.parameters()).device

        for _ in range(max_new_tokens):
            # 输入最近 max_seq_len 个 token（超长则截断）
            input_ids = torch.tensor([generated[-self.max_seq_len:]], device=device)
            logits = self.forward(input_ids)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :] / temperature

            if top_k is not None:
                # Top-k 采样
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

        return generated


# -------------------- 5. 数据准备（字符级）--------------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y


# -------------------- 6. 训练函数 --------------------
def train(model, dataloader, optimizer, device, max_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (batch, seq_len, vocab_size)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:2d}/{max_epochs}, Loss = {avg_loss:.4f}")


# -------------------- 7. 主程序 --------------------
def main():
    # 超参数
    SEQ_LEN = 64          # 上下文长度
    D_MODEL = 128         # 嵌入维度
    NUM_HEADS = 4         # 注意力头数
    NUM_LAYERS = 4        # 解码器层数
    D_FF = 512            # 前馈网络维度
    DROPOUT = 0.1
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001

    # 训练文本（使用一段英文诗歌，也可换成任何文本）
    text = (
        "The sun does arise,\n"
        "And make happy the skies;\n"
        "The merry bells ring\n"
        "To welcome the Spring;\n"
        "The skylark and thrush,\n"
        "The birds of the bush,\n"
        "Sing louder around,\n"
        "To the bells' cheerful sound;\n"
        "While our sports shall be seen\n"
        "On the echoing green.\n"
    )
    # 重复几次让数据多一些
    text = text * 20

    # 创建数据集和数据加载器
    dataset = CharDataset(text, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalLanguageModel(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_len=SEQ_LEN
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("开始训练...")
    train(model, dataloader, optimizer, device, max_epochs=EPOCHS)

    # 文本生成
    start_char = 'T'   # 起始字符
    start_token = dataset.stoi[start_char]
    generated_ids = model.generate(start_token, max_new_tokens=100, temperature=0.8, top_k=10)
    generated_text = ''.join([dataset.itos[idx] for idx in generated_ids])
    print("\n生成的文本:\n")
    print(generated_text)


if __name__ == "__main__":
    main()