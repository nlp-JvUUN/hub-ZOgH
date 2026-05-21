"""
训练基于transformer的单向语言模型，并完成文本生成。
1、训练模型 (作业1)
2、文本生成 (作业2)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import random
import math

# ===================== 数据 =====================
def load_corpus(path):
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        texts.append(f.read())
    return "".join(texts)

def build_vocab(data):
    texts = sorted(set(data))
    idx2char = {i:c for i, c in enumerate(texts)}
    char2idx = {c:i for i, c in idx2char.items()}
    return idx2char, char2idx

class TextDataSet(Dataset):
    def __init__(self, text, char2id, seq_len):
        super().__init__()
        self.seq_len = seq_len
        ids = [char2id[c] for c in text if c in char2id]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return (
            self.data[index: index + self.seq_len],
            self.data[index + 1: index + self.seq_len + 1]
        )

# ===================== 模型 =====================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, dim_feedforward, num_head, num_layers, dropout):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            activation=nn.functional.gelu,
            batch_first=True,
        )
        norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch, seq_len = x.shape
        we = self.word_embedding(x)     # (batch, seq_len, embed_dim)
        # 位置编码：生成 0..seq_len-1 的索引，并扩展 batch 维度
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pe = self.position_embedding(pos_ids) # (1, seq_len, embed_dim)
        embed_x = we + pe
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        transformer_x = self.transformer.forward(embed_x, mask=mask)   # (batch, seq_len, embed_dim)
        logits = self.output(transformer_x)                       # (B, T, vocab_size)
        return logits


# ===================== 训练、验证 =====================
def run_epoch(model, loader, criterion, optimizer, device, train=False):
    total_loss = 0.0
    total_tokens = 0
    if train:
        model.train()
    else:
        model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ===================== 文本生成 =====================
def generate(model, start_text, char2idx, idx2char, max_len=100, temperature=1.0, top_k=None):
    """基于训练好的模型生成文本"""
    model.eval()
    device = next(model.parameters()).device
    
    # 将起始文本转为索引
    if start_text:
        input_ids = [char2idx.get(c, random.choice(list(char2idx.values()))) for c in start_text]
    else:
        input_ids = [random.choice(list(char2idx.values()))]
    
    generated = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_len):
            # 取最后 seq_len 个字符
            x = torch.tensor([generated[-model.position_embedding.num_embeddings:]], dtype=torch.long).to(device)
            
            # 前向传播
            logits = model(x)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)
            
            # Top-k 采样
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
    
    # 转换回字符
    result = "".join([idx2char.get(i, "") for i in generated])
    return result


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    
    model = TransformerModel(
        vocab_size=len(checkpoint["char2idx"]),
        seq_len=args["seq_len"],
        embed_dim=args["embed_dim"],
        dim_feedforward=args["ff_dim"],
        num_head=args["num_head"],
        num_layers=args["num_layers"],
        dropout=args["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    
    return model, checkpoint["char2idx"], checkpoint["idx2char"]


# ===================== 主函数 =====================
def main():
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="模型 checkpoint 路径")
    parser.add_argument("--start_text", type=str, default="", help="生成起始文本")
    parser.add_argument("--max_len", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k 采样")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--num_head", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--save", default="best_model.pt")
    args = parser.parse_args()

    # 设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # ===================== 生成模式 =====================
    if args.mode == "generate":
        if not os.path.exists(args.checkpoint):
            print(f"错误: 找不到模型文件 {args.checkpoint}")
            return
        model, char2idx, idx2char = load_model(args.checkpoint, device)
        print(f"模型加载成功! 词表大小: {len(char2idx)}")
        
        print(f"\n起始文本: {args.start_text}")
        generated_text = generate(
            model, args.start_text, char2idx, idx2char,
            max_len=args.max_len, temperature=args.temperature, top_k=args.top_k
        )
        print(f"生成结果: {generated_text}")
        return

    # ===================== 训练模式 =====================
    # 数据路径 - 使用脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "corpus.txt")
    text = load_corpus(path)
    if not text:
        raise FileNotFoundError("语料不存在")
    
    # 构建词表
    idx2char, char2idx = build_vocab(text)
    vocab_size = len(idx2char)
    print(f"总词表:{vocab_size}")

    # 数据分割(训练集、验证集)
    texts = text.splitlines()
    random.shuffle(texts)
    split = int(len(texts) * (1 - args.val_ratio))
    train_text = "\n".join(texts[:split])
    val_text = "\n".join(texts[split:])

    train_data = TextDataSet(train_text, char2idx, args.seq_len)
    val_data = TextDataSet(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True)

    # 模型
    model = TransformerModel(
        vocab_size=vocab_size, 
        seq_len=args.seq_len,
        embed_dim=args.embed_dim, 
        dim_feedforward=args.ff_dim, 
        num_head=args.num_head, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数:{total_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # 训练过程
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'LR':>12}")
    print("-" * 70)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_loss, val_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {val_loss:>10.4f}  {val_ppl:>10.2f}  {current_lr:>12.6f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

if __name__ == "__main__":
    main()
