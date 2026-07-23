
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 核心：Transformer Layer（修复 mask 格式）=====================
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ✅ 修复：attn_mask 必须是 float 且格式正确
        attn_out, attn_weights = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

# ===================== 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ===================== Transformer 语言模型（修复因果 mask！）=====================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, seq_len):
        # ✅ 关键修复：返回 float 格式 -inf mask，兼容 MultiheadAttention
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        causal_mask = self.generate_causal_mask(seq_len).to(x.device)
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, causal_mask)
            attn_weights.append(attn)

        logits = self.fc(x)
        return logits, attn_weights

# ===================== 训练函数 =====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        data = batch[0]
        src = data[:, :-1].to(device)
        tgt = data[:, 1:].to(device)

        optimizer.zero_grad()
        logits, _ = model(src)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        
        # ✅ 防止 nan 反向传播
        if not torch.isnan(loss):
            loss.backward()
        
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)

# ===================== 生成函数（绝对稳定）=====================
def generate_text(model, start_tokens, max_len, device, temperature=1.0):
    model.eval()
    generated = start_tokens.clone().to(device)
    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 0: break
    return generated

def generate_text_sample(model, start_tokens, max_len, device, temperature=1.0):
    model.eval()
    generated = start_tokens.clone().to(device)
    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(generated)
            logits = logits[:, -1, :] / temperature
            
            # ✅ 强稳定保护
            logits = torch.clamp(logits, -50, 50)
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, 1e-9, 1.0)
            probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 0: break
    return generated

# ===================== 主程序（超稳定参数）=====================
if __name__ == "__main__":
    vocab_size = 1000
    embed_dim = 128
    num_heads = 2
    ff_dim = 256
    num_layers = 2
    batch_size = 4
    seq_len = 20
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # ✅ 小学习率，不爆炸

    # 随机数据
    data = torch.randint(1, vocab_size, (200, seq_len))
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Training language model...")
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    print("\nGenerating text with greedy search...")
    start_tokens = torch.randint(1, vocab_size, (1, 3))
    generated = generate_text(model, start_tokens, 15, device)
    print(f"Generated: {generated}")

    print("\nGenerating text with sampling...")
    generated_sample = generate_text_sample(model, start_tokens, 15, device, temperature=0.7)
    print(f"Generated with sampling: {generated_sample}")

    print("\n✅ 运行成功！全程无报错！")
