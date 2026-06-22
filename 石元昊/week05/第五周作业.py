"""
第五周作业：基于Transformer的单向语言模型训练与文本生成
训练数据：毛泽东选集
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time

# ===================== 配置参数 =====================
class Config:
    # 数据相关
    data_path = r"E:\badou\毛泽东选集\毛泽东选集_完整合集.txt"
    seq_length = 64          # 序列长度
    batch_size = 32          # 批次大小
    
    # 模型相关
    d_model = 128            # 词嵌入维度
    n_heads = 4              # 注意力头数
    n_layers = 4             # Transformer层数
    d_ff = 256               # 前馈网络维度
    dropout = 0.1            # dropout率
    
    # 训练相关
    epochs = 50              # 训练轮数
    learning_rate = 0.001    # 学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成相关
    gen_length = 200         # 生成文本长度
    temperature = 0.8        # 生成温度（越低越确定，越高越随机）

# ===================== 数据处理 =====================
class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, text, seq_length, vocab, char_to_idx):
        self.seq_length = seq_length
        self.vocab = vocab
        self.char_to_idx = char_to_idx
        
        # 将文本转换为索引序列
        self.data = [char_to_idx[char] for char in text if char in char_to_idx]
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        # 输入序列
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        # 目标序列（向后偏移一位，单向语言模型）
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


class TextProcessor:
    """文本处理器"""
    def __init__(self):
        self.vocab = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text):
        """构建词汇表"""
        # 去重并排序字符
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"文本总长度: {len(text)} 字符")
        
    def encode(self, text):
        """将文本编码为索引"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices):
        """将索引解码为文本"""
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])


# ===================== Transformer模型 =====================
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLanguageModel(nn.Module):
    """基于Transformer的单向语言模型"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器层（使用causal mask实现单向）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（causal mask），确保只能看到前面的token"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src):
        """
        src: (batch, seq_len) 输入序列
        """
        # 词嵌入 + 位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 生成因果掩码
        mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Transformer编码
        output = self.transformer_encoder(src, mask=mask)
        
        # 输出层
        output = self.fc(output)
        
        return output  # (batch, seq_len, vocab_size)
    
    @torch.no_grad()
    def generate(self, start_text, processor, gen_length=100, temperature=1.0):
        """文本生成"""
        self.eval()
        
        # 编码起始文本
        input_indices = processor.encode(start_text)
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(Config.device)
        
        generated = input_indices.copy()
        
        for _ in range(gen_length):
            # 取最近的一段序列（不超过seq_length）
            if len(generated) > Config.seq_length:
                generated = generated[-Config.seq_length:]
            
            input_tensor = torch.tensor([generated], dtype=torch.long).to(Config.device)
            
            # 前向传播
            output = self.forward(input_tensor)
            
            # 取最后一个token的输出
            logits = output[0, -1, :] / temperature
            
            # 使用softmax转换为概率
            probs = torch.softmax(logits, dim=-1)
            
            # 根据概率采样下一个token
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_idx)
        
        # 解码为文本
        return processor.decode(generated)


# ===================== 训练函数 =====================
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    """训练模型"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            output = model(src)
            
            # 计算损失
            # output: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # tgt: (batch, seq_len) -> (batch * seq_len)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每10个batch打印一次
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        
        # 学习率调度
        scheduler.step()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 每5个epoch保存一次模型并生成示例文本
        if (epoch + 1) % 5 == 0:
            save_model(model, epoch + 1, avg_loss)
            generate_sample(model, processor, epoch + 1)


def save_model(model, epoch, loss):
    """保存模型"""
    save_path = r"E:\badou\transformer_lang_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'config': {
            'vocab_size': processor.vocab_size,
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_layers': Config.n_layers,
            'd_ff': Config.d_ff,
        }
    }, save_path)
    print(f"模型已保存到: {save_path}")


def generate_sample(model, processor, epoch):
    """生成示例文本"""
    model.eval()
    start_texts = ["中国", "革命", "人民", "我们"]
    
    print(f"\n--- Epoch {epoch} 生成示例 ---")
    for start_text in start_texts:
        generated = model.generate(
            start_text=start_text,
            processor=processor,
            gen_length=50,
            temperature=Config.temperature
        )
        print(f"[{start_text}] -> {generated[:100]}...")
    print("--- 示例结束 ---\n")


# ===================== 主函数 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("第五周作业：基于Transformer的单向语言模型")
    print("=" * 60)
    
    # 读取训练数据
    print("\n[1] 读取训练数据...")
    with open(Config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"数据路径: {Config.data_path}")
    print(f"文本长度: {len(text)} 字符")
    print(f"文本前100字符: {text[:100]}...")
    
    # 处理文本
    print("\n[2] 处理文本，构建词汇表...")
    processor = TextProcessor()
    processor.build_vocab(text)
    
    # 创建数据集和数据加载器
    print("\n[3] 创建数据集...")
    dataset = TextDataset(text, Config.seq_length, processor.vocab, processor.char_to_idx)
    train_loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    print(f"数据集大小: {len(dataset)} 样本")
    print(f"批次数量: {len(train_loader)}")
    
    # 创建模型
    print("\n[4] 创建Transformer语言模型...")
    model = TransformerLanguageModel(
        vocab_size=processor.vocab_size,
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        n_layers=Config.n_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout
    ).to(Config.device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    print(f"使用设备: {Config.device}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 开始训练
    print("\n[5] 开始训练...")
    print("=" * 60)
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=Config.epochs,
        device=Config.device
    )
    
    # 最终生成示例
    print("\n[6] 最终文本生成测试...")
    print("=" * 60)
    start_texts = ["中国共产党", "中国人民", "革命战争", "马克思主义"]
    
    for start_text in start_texts:
        generated = model.generate(
            start_text=start_text,
            processor=processor,
            gen_length=Config.gen_length,
            temperature=Config.temperature
        )
        print(f"\n输入: {start_text}")
        print(f"生成: {generated}")
        print("-" * 60)
    
    print("\n训练完成！")
    print(f"模型保存路径: E:\badou\transformer_lang_model.pth")
