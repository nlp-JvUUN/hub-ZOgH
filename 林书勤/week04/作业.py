import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# BERT实现 - 核心结构演示
test_sentence = "deep learning model"  # 英文示例
vocab = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
for char in test_sentence.replace(" ", ""):
    if char not in vocab:
        vocab[char] = len(vocab)

# Tokenization
tokens = ['[CLS]'] + [c for c in test_sentence.replace(" ", "")] + ['[SEP]']
token_ids = [vocab.get(t, 1) for t in tokens]
pos_ids = list(range(len(token_ids)))

# 超参数
EMBED_DIM = 64    
NUM_HEADS = 4     
HEAD_DIM = 16     
FF_DIM = 128      
VOCAB_SIZE = len(vocab)
SEQ_LEN = len(token_ids)

class SimpleBERT(nn.Module):
    """- 单Transformer层"""
    def __init__(self):
        super().__init__()
        # 嵌入层
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(512, EMBED_DIM)
        
        # 注意力线性变换
        self.W_q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_v = nn.Linear(EMBED_DIM, EMBED_DIM)
        
        # 前馈网络
        self.ff1 = nn.Linear(EMBED_DIM, FF_DIM)
        self.ff2 = nn.Linear(FF_DIM, EMBED_DIM)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        
        # 输出层
        self.out = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        
    def attention(self, Q, K, V):
        """多头注意力计算"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
    
    def forward(self, token_ids, pos_ids):
        # 嵌入
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)
        residual = x
        
        # 注意力
        Q = self.W_q(x).view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        K = self.W_k(x).view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        V = self.W_v(x).view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        
        Q = Q.permute(1, 0, 2)  # [heads, seq, dim]
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        
        attn_out = self.attention(Q, K, V)
        attn_out = attn_out.permute(1, 0, 2).reshape(SEQ_LEN, EMBED_DIM)
        
        # 残差连接+层归一化
        x = self.ln1(residual + attn_out)
        residual = x
        
        # 前馈网络
        x = F.gelu(self.ff1(x))
        x = self.ff2(x)
        
        # 最终输出
        x = self.ln2(residual + x)
        return self.out(x)

# 准备输入
model = SimpleBERT()
inputs = torch.tensor(token_ids).unsqueeze(0)  # 增加batch维度
positions = torch.tensor(pos_ids).unsqueeze(0)

# 前向传播
outputs = model(inputs, positions)
print("模型输出形状:", outputs.shape)  # [1, seq_len, vocab_size]

# 简单训练示例
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟训练步骤
for epoch in range(3):  # 仅演示3轮
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    logits = model(inputs, positions)
    
    # 创建简单标签（示例：预测下一个字符）
    targets = inputs[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    # 计算损失
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 预测示例
model.eval()
with torch.no_grad():
    probs = F.softmax(outputs[0], dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    print("\n预测结果:")
    for token, pred in zip(tokens, preds):
        pred_word = list(vocab.keys())[list(vocab.values()).index(pred.item())]
        print(f"  {token} -> {pred_word}")

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
