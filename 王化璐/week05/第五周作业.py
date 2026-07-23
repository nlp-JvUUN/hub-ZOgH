import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== 配置：适配联想小新的超小参数 =====================
DEVICE = torch.device("cpu")
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
SEQ_LEN = 32
BATCH_SIZE = 32
LR = 1e-3
CORPUS_PATH = r"E:\AI课学习\week4语言模型\week4 语言模型\循环神经网络语言模型\corpus.txt"
SAVE_PATH = "mini_transformer_lm.pt"

# ===================== 1. 因果掩码 + 位置编码 =====================
def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

# ===================== 2. 数据加载与词表 =====================
def load_corpus(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

# ===================== 3. Transformer 单向语言模型 =====================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, DROPOUT)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=4*EMBED_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(EMBED_DIM)
        x = self.pos_encoder(x)
        mask = create_causal_mask(seq_len, x.device)
        memory = torch.zeros_like(x)
        x = self.decoder(x, memory, tgt_mask=mask)
        return self.fc(x)

# ===================== 4. 训练与评估 =====================
def run_epoch(model, loader, criterion, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

# ===================== 5. 4种文本生成策略 =====================
@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_len=50):
    strategies = {
        "贪心解码": "greedy",
        "温度采样": "temperature",
        "Top-K采样": "topk",
        "束搜索": "beam"
    }
    results = {}
    model.eval()

    for name, mode in strategies.items():
        tokens = [char2idx[c] for c in prompt if c in char2idx]
        if not tokens: continue
        gen = tokens.copy()

        if mode == "greedy":
            for _ in range(max_len):
                inp = torch.tensor(gen[-32:], device=DEVICE).unsqueeze(0)
                next_token = model(inp)[0, -1].argmax().item()
                gen.append(next_token)

        elif mode == "temperature":
            for _ in range(max_len):
                inp = torch.tensor(gen[-32:], device=DEVICE).unsqueeze(0)
                logits = model(inp)[0, -1] / 0.7
                prob = F.softmax(logits, dim=-1)
                gen.append(torch.multinomial(prob, 1).item())

        elif mode == "topk":
            for _ in range(max_len):
                inp = torch.tensor(gen[-32:], device=DEVICE).unsqueeze(0)
                logits = model(inp)[0, -1]
                v, idx = torch.topk(logits, 10)
                gen.append(idx[torch.multinomial(F.softmax(v, -1), 1)].item())

        elif mode == "beam":
            beams = [(gen, 0.0)]
            for _ in range(max_len):
                new_beams = []
                for seq, s in beams:
                    inp = torch.tensor(seq[-32:], device=DEVICE).unsqueeze(0)
                    log_prob = F.log_softmax(model(inp)[0, -1], -1)
                    topv, topi = torch.topk(log_prob, 2)
                    for i in range(2):
                        new_beams.append((seq + [topi[i].item()], s + topv[i].item()))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:2]
            gen = beams[0][0]

        results[name] = "".join([idx2char[i] for i in gen])
    return results

# ===================== 主函数：一键训练+生成 =====================
if __name__ == "__main__":
    # 1. 加载数据
    print("加载语料...")
    text = load_corpus(CORPUS_PATH)
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"字符数：{len(text)} | 词表大小：{vocab_size}")

    # 2. 数据集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines)*0.95)
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, SEQ_LEN)
    val_ds = CharDataset(val_text, char2idx, SEQ_LEN)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=True)

    # 3. 模型
    model = TransformerLM(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_ppl = float("inf")

    # 4. 训练
    print("\n开始训练（CPU微型模型，速度很快）...")
    print(f"Epoch | Train Loss | Train PPL | Val Loss | Val PPL")
    print("-"*50)

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion)

        if va_ppl < best_ppl:
            best_ppl = va_ppl
            torch.save({
                "model": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char
            }, SAVE_PATH)

        print(f"{epoch:2d} | {tr_loss:.4f} | {tr_ppl:.2f} | {va_loss:.4f} | {va_ppl:.2f}")

    # 5. 自动生成文本
    print("\n训练完成！测试文本生成：")
    prompts = ["深度学习", "人工智能", "语言模型"]
    for p in prompts:
        print(f"\n【前缀】{p}")
        res = generate(model, char2idx, idx2char, p)
        for k, v in res.items():
            print(f"  {k}：{v}")
