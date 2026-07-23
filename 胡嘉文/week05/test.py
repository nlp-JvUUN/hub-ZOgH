"""
测试训练好的GPT模型（Decoder-only架构），进行文本生成
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

# 超参数设置（与训练时保持一致）
batch_size = 64
seq_len = 32
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_heads = 4
dropout = 0.1

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用CUDA GPU")
else:
    device = torch.device("cpu")
    print("使用CPU")

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
        
        return output
    
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

def generate_text(model, vocab, start_text, max_length=200, temperature=1.0):
    """
    使用训练好的GPT模型生成文本
    """
    model.eval()
    vocab_reverse = {v: k for k, v in vocab.items()}
    
    # 准备起始序列
    current_text = list(start_text)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 将当前文本转换为索引
            input_indices = []
            for char in current_text[-seq_len:]:
                if char in vocab:
                    input_indices.append(vocab[char])
                else:
                    input_indices.append(vocab["<UNK>"])
            
            # 转换为张量 [seq_len, 1]
            input_tensor = torch.LongTensor(input_indices).unsqueeze(1).to(device)
            
            # 预测下一个字符
            output = model(input_tensor)
            logits = output[-1, 0, :] / temperature
            
            # 采样下一个字符
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = vocab_reverse.get(next_idx, "<UNK>")
            
            # 添加到当前文本
            current_text.append(next_char)
    
    return ''.join(current_text)

def main():
    # 1. 加载语料和构建词汇表
    corpus_path = "/Users/hjw/Project/nlp-study/胡嘉文/week05/corpus.txt"
    text = load_corpus(corpus_path)
    vocab = build_vocab(text)
    print(f"词汇表大小: {len(vocab)}")
    
    # 2. 加载GPT模型（Decoder-only架构）
    model = GPTModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(torch.load("/Users/hjw/Project/nlp-study/胡嘉文/week05/best_model.pt"))
    print("GPT模型(Decoder-only)加载成功!")
    
    # 3. 测试文本生成
    print("\n" + "="*50)
    print("测试1: 以'黄金'开头生成文本")
    print("="*50)
    start_text1 = "黄金"
    generated_text1 = generate_text(model, vocab, start_text1, max_length=300, temperature=0.8)
    print(generated_text1)
    
    print("\n" + "="*50)
    print("测试2: 以'期货'开头生成文本")
    print("="*50)
    start_text2 = "期货"
    generated_text2 = generate_text(model, vocab, start_text2, max_length=300, temperature=0.9)
    print(generated_text2)
    
    print("\n" + "="*50)
    print("测试3: 以'股票'开头生成文本")
    print("="*50)
    start_text3 = "股票"
    generated_text3 = generate_text(model, vocab, start_text3, max_length=300, temperature=0.7)
    print(generated_text3)

if __name__ == "__main__":
    main()
