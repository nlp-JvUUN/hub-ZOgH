"""
训练基于Transformer的单向语言模型
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_corpus(file_pattern="*.txt"):
    """
    从多个文本文件中加载语料库
    
    参数:
        file_pattern: 文件模式，用于匹配要加载的文本文件
        
    返回:
        所有文本文件内容拼接后的字符串
    """
    texts = []
    for file_path in glob.glob(file_pattern):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                texts.append(file.read())
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    return "".join(texts)


def build_vocab(text):
    """
    从文本构建字符到索引的映射词典
    
    参数:
        text: 输入的文本字符串
        
    返回:
        char_to_idx: 字符到索引的映射字典
        idx_to_char: 索引到字符的映射字典
    """
    # 获取所有唯一的字符并排序
    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char


class CharDataset(Dataset):
    """
    字符级数据集类，用于创建训练样本
    
    每个样本包含长度为seq_len的输入序列和对应的标签序列
    标签序列是输入序列向右移动一个位置的序列
    """
    
    def __init__(self, text, char_to_idx, seq_len):
        """
        初始化数据集
        
        参数:
            text: 文本字符串
            char_to_idx: 字符到索引的映射字典
            seq_len: 序列长度
        """
        self.seq_len = seq_len
        
        # 将文本转换为索引序列
        self.data = torch.tensor([char_to_idx.get(char, 0) for char in text], dtype=torch.long)
    
    def __len__(self):
        """返回数据集中的样本数量"""
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: 样本索引
            
        返回:
            x: 输入序列
            y: 目标序列（输入序列向右移动一位）
        """
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为输入序列添加位置信息，使Transformer能够感知序列中字符的位置
    使用正弦和余弦函数生成位置编码
    """
    
    def __init__(self, embedding_dim, dropout=0.1, max_len=1000):
        """
        初始化位置编码模块
        
        参数:
            embedding_dim: 嵌入维度
            dropout: dropout概率
            max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵
        position_encoding = torch.zeros(max_len, embedding_dim)
        
        # 生成位置索引
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        # 计算频率项
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * 
            -(math.log(10000.0) / embedding_dim)
        )
        
        # 应用正弦和余弦函数
        position_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        position_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer("position_encoding", position_encoding)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, seq_len, embedding_dim)
            
        返回:
            添加了位置编码的输入张量
        """
        # 添加位置编码
        x = x + self.position_encoding[:x.size(1), :]
        return self.dropout(x)


def generate_square_subsequent_mask(size, device):
    """
    生成因果注意力掩码（上三角矩阵）
    
    确保在生成时，每个位置只能看到之前的位置，实现单向语言模型
    
    参数:
        size: 掩码大小
        device: 设备
        
    返回:
        形状为(size, size)的上三角掩码矩阵
    """
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class TransformerLM(nn.Module):
    """
    Transformer语言模型
    
    基于Transformer架构的单向语言模型，适用于字符级文本生成
    """
    
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, dropout):
        """
        初始化模型
        
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏层维度
            num_layers: Transformer编码器层数
            dropout: dropout概率
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",  # 使用GELU激活函数
            batch_first=True,   # 输入形状为(batch, seq, feature)
        )
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 保存嵌入维度用于缩放
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列，形状为(batch_size, seq_len)
            
        返回:
            logits: 输出logits，形状为(batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        embeddings = self.embedding(x) * math.sqrt(self.embedding_dim)
        
        # 添加位置编码
        embeddings = self.positional_encoding(embeddings)
        
        # 生成因果注意力掩码
        mask = generate_square_subsequent_mask(x.size(1), x.device)
        
        # 通过Transformer编码器
        transformer_output = self.transformer(embeddings, mask=mask)
        
        # 应用dropout
        transformer_output = self.dropout(transformer_output)
        
        # 计算输出logits
        logits = self.output_layer(transformer_output)
        
        return logits


def run_epoch(model, data_loader, criterion, optimizer, device, is_training=True):
    """
    运行一个训练或验证周期
    
    参数:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        is_training: 是否为训练模式
        
    返回:
        average_loss: 平均损失
        perplexity: 困惑度
    """
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch_inputs, batch_targets in data_loader:
        # 移动到设备
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # 前向传播
        logits = model(batch_inputs)
        
        # 计算损失
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # 展平成(batch*seq_len, vocab_size)
            batch_targets.reshape(-1)              # 展平成(batch*seq_len)
        )
        
        # 训练模式下进行反向传播和优化
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # 累计损失和token数
        total_loss += loss.item() * batch_targets.numel()
        total_tokens += batch_targets.numel()
    
    # 计算平均损失和困惑度
    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    
    return average_loss, perplexity


def main():
    """主函数，控制整个训练流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练Transformer单向语言模型")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--seq_len", type=int, default=64, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--embedding_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=256, help="前馈网络隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=2, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout概率")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    
    # 数据参数
    parser.add_argument("--corpus", default="*.txt", help="语料库文件模式")
    parser.add_argument("--save", default="transformer_lm.pt", help="模型保存路径")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型架构: Transformer语言模型")
    print("-" * 60)
    
    # 加载语料库
    print("加载语料库...")
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到匹配模式 '{args.corpus}' 的文件")
    print(f"语料库字符数: {len(text):,}")
    
    # 构建词汇表
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"词汇表大小: {vocab_size}")
    
    # 分割训练集和验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split_index = int(len(lines) * (1 - args.val_ratio))
    
    train_text = "\n".join(lines[:split_index])
    val_text = "\n".join(lines[split_index:])
    
    print(f"训练集行数: {len(lines[:split_index]):,}")
    print(f"验证集行数: {len(lines[split_index:]):,}")
    print("-" * 60)
    
    # 创建数据集和数据加载器
    train_dataset = CharDataset(train_text, char_to_idx, args.seq_len)
    val_dataset = CharDataset(val_text, char_to_idx, args.seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 60)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    print("开始训练...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 62)
    
    best_val_perplexity = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # 训练周期
        train_loss, train_perplexity = run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        
        # 验证周期
        with torch.no_grad():
            val_loss, val_perplexity = run_epoch(
                model, val_loader, criterion, optimizer, device, is_training=False
            )
        
        # 保存最佳模型
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            torch.save({
                "model_state_dict": model.state_dict(),
                "char_to_idx": char_to_idx,
                "idx_to_char": idx_to_char,
                "args": vars(args),
                "val_perplexity": val_perplexity,
            }, args.save)
            
            # 标记最佳模型
            marker = "  *"
        else:
            marker = ""
        
        # 打印训练进度
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_perplexity:>10.2f}  "
              f"{val_loss:>10.4f}  {val_perplexity:>10.2f}{marker}")
    
    print("-" * 62)
    print(f"训练完成!")
    print(f"最佳验证困惑度: {best_val_perplexity:.2f}")
    print(f"模型已保存至: {args.save}")
    
    return model, char_to_idx, idx_to_char


if __name__ == "__main__":
    main()
