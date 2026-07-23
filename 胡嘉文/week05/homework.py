"""
训练基于Transformer Decoder-only的语言模型（类似GPT架构），并完成文本生成。
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from collections import Counter

# 超参数设置
batch_size = 64
seq_len = 32
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_heads = 4
dropout = 0.1
epochs = 10
lr = 1e-4

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用MPS (Apple Silicon GPU) 进行加速训练")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用CUDA GPU进行加速训练")
else:
    device = torch.device("cpu")
    print("使用CPU进行训练")

class CorpusDataset(Dataset):
    """语料数据集"""
    def __init__(self, text, vocab, seq_len):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = self.tokenize(text)
        
    def tokenize(self, text):
        """将文本转换为token序列"""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab["<UNK>"])
        return tokens
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        inputs = self.data[idx:idx+self.seq_len]
        targets = self.data[idx+1:idx+self.seq_len+1]
        return torch.LongTensor(inputs), torch.LongTensor(targets)

class GPTModel(nn.Module):
    """基于Transformer Decoder-only的语言模型（类似GPT架构）"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        super(GPTModel, self).__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer Encoder层（使用EncoderLayer实现Decoder-only）
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        self.embedding_dim = embedding_dim
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: 输入序列 [seq_len, batch_size]
            src_mask: 因果掩码，防止看到未来的token
        Returns:
            output: [seq_len, batch_size, vocab_size]
        """
        # Embedding + 位置编码
        emb = self.embedding(src) * np.sqrt(self.embedding_dim)
        emb = self.pos_encoder(emb)
        
        # 如果没有提供掩码，生成因果掩码
        if src_mask is None:
            src_mask = self.generate_causal_mask(src.size(0)).to(device)
        
        # Transformer Encoder处理（使用因果掩码实现Decoder-only）
        output = self.transformer_encoder(emb, mask=src_mask)
        
        # 输出层
        output = self.fc(output)
        
        return output  # [seq_len, batch_size, vocab_size]
    
    def generate_causal_mask(self, sz):
        """生成因果掩码，防止模型看到未来的token"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def build_vocab(text):
    """构建词汇表"""
    counter = Counter(text)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, _ in counter.most_common():
        vocab[char] = len(vocab)
    return vocab

def load_corpus(file_path):
    """加载语料文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def train():
    # 加载语料
    corpus_path = "/Users/hjw/Project/nlp-study/胡嘉文/week05/corpus.txt"
    text = load_corpus(corpus_path)
    print(f"语料长度: {len(text)}")
    
    # 构建词汇表
    vocab = build_vocab(text)
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建数据集和数据加载器
    dataset = CorpusDataset(text, vocab, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建GPT模型（Decoder-only架构）
    model = GPTModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    print("使用Decoder-only架构（GPT风格）")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 转换为[seq_len, batch_size]格式
            inputs = inputs.transpose(0, 1).contiguous().to(device)
            targets = targets.transpose(0, 1).contiguous().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/Users/hjw/Project/nlp-study/胡嘉文/week05/best_model.pt")
            print(f"模型已保存，当前最佳损失: {best_loss:.4f}")

if __name__ == "__main__":
    train()
