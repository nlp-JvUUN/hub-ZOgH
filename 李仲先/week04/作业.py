"""
Transformer 层的 PyTorch 实现
包含多头注意力机制、前馈网络、完整的 Transformer 层
以及简单的分类任务训练和验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头的数量
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # Q、K、V 的线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)  # 输出线性层
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的注意力掩码
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # 线性变换后重塑为多头形式: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（用于屏蔽填充位置或未来位置）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 注意力权重与 V 相乘
        out = torch.matmul(attn_weights, v)
        # 拼接多头输出: (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(out)


class FeedForward(nn.Module):
    """前馈神经网络（Feed-Forward Network）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 隐藏层维度（通常为 d_model 的 4 倍）
            dropout: Dropout 概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """前向传播: Linear -> GELU -> Dropout -> Linear"""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """完整的 Transformer 编码器层
    
    结构: 多头注意力 -> 残差连接 + LayerNorm -> 前馈网络 -> 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化 Transformer 层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # 注意力后的 LayerNorm
        self.norm2 = nn.LayerNorm(d_model)  # 前馈网络后的 LayerNorm
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的注意力掩码
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 多头注意力 + 残差连接 + LayerNorm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SimpleTransformer(nn.Module):
    """简单的 Transformer 分类模型
    
    结构: 词嵌入 + 位置编码 -> 多层 Transformer -> 平均池化 -> 分类
    """
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, dropout=0.1):
        """
        初始化 Transformer 模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            num_layers: Transformer 层数
            num_classes: 分类类别数
            dropout: Dropout 概率
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 可学习的位置编码（最大长度 512）
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        # 堆叠的 Transformer 层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # 分类输出层
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 token IDs，形状为 (batch_size, seq_len)
            mask: 可选的注意力掩码
        
        Returns:
            分类 logits，形状为 (batch_size, num_classes)
        """
        seq_len = x.size(1)
        # 词嵌入 + 位置编码
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过多层 Transformer
        for layer in self.layers:
            x = layer(x, mask)
        
        # 平均池化: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.mean(dim=1)
        return self.fc_out(x)


def create_dummy_data(num_samples, seq_len, vocab_size, num_classes):
    """创建随机数据用于演示"""
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()  # 清零梯度
        outputs = model(batch_x)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def main():
    """主函数：设置超参数、创建模型、训练和验证"""
    torch.manual_seed(42)  # 设置随机种子以复现结果
    
    # 模型超参数
    vocab_size = 1000   # 词汇表大小
    d_model = 128        # 模型维度
    num_heads = 8        # 注意力头数量
    d_ff = 512           # 前馈网络隐藏层维度
    num_layers = 2       # Transformer 层数
    num_classes = 5      # 分类类别数
    dropout = 0.1        # Dropout 概率
    
    # 训练超参数
    batch_size = 32
    seq_len = 20
    num_train_samples = 500
    num_val_samples = 100
    num_epochs = 10
    learning_rate = 0.001
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建训练和验证数据
    train_x, train_y = create_dummy_data(num_train_samples, seq_len, vocab_size, num_classes)
    val_x, val_y = create_dummy_data(num_val_samples, seq_len, vocab_size, num_classes)
    
    # 创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = SimpleTransformer(vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, dropout)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {num_train_samples}, Validation samples: {num_val_samples}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print("\nStarting training...\n")
    
    # 训练循环
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    print("\nTraining completed!")
    
    # 测试预测
    test_sample = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    model.eval()
    with torch.no_grad():
        output = model(test_sample)
        prediction = output.argmax(dim=1).item()
    print(f"\nTest prediction: Class {prediction}")


if __name__ == "__main__":
    main()