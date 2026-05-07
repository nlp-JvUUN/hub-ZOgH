"""
第三周作业.py
多分类任务：检测"你"字在 5 字文本中的位置

任务：输入 5 个字的文本，包含"你"字 → 输出"你"在第几位（1-5 类）
模型：Embedding → RNN/LSTM → 池化 → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 10000  # 总样本数（增加样本）
TEXT_LEN = 5       # 固定 5 个字
NUM_CLASSES = 5    # 5 个类别（"你"在第 1-5 位）
EMBED_DIM = 64
HIDDEN_DIM = 128
LR = 2e-3          # 稍微增加学习率
BATCH_SIZE = 64
EPOCHS = 30        # 增加训练轮次
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字（不包含"你"）
COMMON_CHARS = [
    '我', '他', '她', '它', '们', '的', '是', '了', '在', '不',
    '有', '这', '个', '那', '要', '就', '都', '而', '及', '与',
    '着', '过', '看', '起', '来', '到', '没', '对', '错', '好',
    '坏', '大', '小', '多', '少', '高', '低', '快', '慢', '上',
    '下', '前', '后', '左', '右', '中', '内', '外', '天', '地',
    '人', '时', '间', '事', '情', '物', '家', '国', '学', '生',
]


def generate_sample(position):
    """
    生成一个样本，"你"字在指定位置
    position: 1-5，表示"你"在第几位
    """
    chars = []
    for i in range(TEXT_LEN):
        if i + 1 == position:
            chars.append('你')
        else:
            chars.append(random.choice(COMMON_CHARS))
    return ''.join(chars), position - 1  # 类别从 0 开始


def build_dataset(n=N_SAMPLES):
    """
    构建数据集，保证每个类别的样本数大致相等
    """
    data = []
    samples_per_class = n // NUM_CLASSES
    
    for pos in range(1, NUM_CLASSES + 1):
        for _ in range(samples_per_class):
            data.append(generate_sample(pos))
    
    # 添加剩余样本
    remaining = n - len(data)
    for _ in range(remaining):
        pos = random.randint(1, NUM_CLASSES)
        data.append(generate_sample(pos))
    
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """
    构建词表
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=TEXT_LEN):
    """
    将文本编码为索引序列
    """
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class RNNClassifier(nn.Module):
    """
    简单 RNN 分类器
    架构：Embedding → RNN → MaxPool → Linear → Softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, 
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(emb)  # (batch, seq_len, hidden_dim)
        
        # Max Pooling：提取每个特征通道的最大值
        pooled = rnn_out.max(dim=1)[0]  # (batch, hidden_dim)
        pooled = self.dropout(pooled)
        
        return self.fc(pooled)  # (batch, num_classes)


class LSTMClassifier(nn.Module):
    """
    LSTM 分类器
    架构：Embedding → LSTM → MaxPool + AvgPool → Linear → Softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向 LSTM
        )
        
        # 双向 LSTM 输出维度×2
        lstm_output_dim = hidden_dim * 2
        
        # 多池化融合
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, hidden_dim),  # ×2 因为两种池化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = F.dropout(emb, p=0.3, training=self.training)
        
        # LSTM 输出
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim*2)
        
        # 转置为 (batch, feature, seq_len) 以便池化
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden_dim*2, seq_len)
        
        # 多池化
        max_pooled = self.max_pool(lstm_out).squeeze(-1)  # (batch, hidden_dim*2)
        avg_pooled = self.avg_pool(lstm_out).squeeze(-1)  # (batch, hidden_dim*2)
        
        # 拼接
        combined = torch.cat([max_pooled, avg_pooled], dim=1)  # (batch, hidden_dim*4)
        
        return self.fc(combined)


class GRUClassifier(nn.Module):
    """
    GRU 分类器（LSTM 的简化版）
    架构：Embedding → GRU → Last Hidden State → Linear → Softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 双向 GRU 输出维度×2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        
        # GRU 输出
        gru_out, h_n = self.gru(emb)  # h_n: (num_layers*2, batch, hidden_dim)
        
        # 拼接双向最后隐藏状态
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim*2)
        
        return self.fc(hidden)


class AttentionLSTMClassifier(nn.Module):
    """
    带 Attention 的 LSTM 分类器
    架构：Embedding → LSTM → Attention → Linear → Softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention 层
        lstm_output_dim = hidden_dim * 2
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        
        # LSTM 输出所有时间步的隐藏状态
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim*2)
        
        # 计算 attention 权重
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # 加权求和
        context = torch.sum(att_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        
        return self.fc(context)


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader, criterion=None):
    """
    评估模型准确率和损失
    """
    model.eval()
    correct = total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)  # (batch, num_classes)
            
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += loss.item()
            
            # 计算准确率
            pred = logits.argmax(dim=1)  # 取概率最大的类别
            correct += (pred == y).sum().item()
            total += len(y)
    
    accuracy = correct / total
    avg_loss = total_loss / len(loader) if criterion is not None else None
    
    return accuracy, avg_loss


def train_model(model_class, model_name, train_loader, val_loader, vocab_size):
    """
    训练单个模型
    """
    print(f"\n{'='*60}")
    print(f"训练模型：{model_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = model_class(vocab_size=vocab_size)
    
    # 损失函数（多分类用 CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  词表大小：{vocab_size}")
    print(f"  模型参数量：{total_params:,}")
    print()
    
    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            # 前向传播
            logits = model(X)  # (batch, num_classes)
            loss = criterion(logits, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        avg_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n最终验证准确率：{best_val_acc:.4f}")
    
    return model, best_val_acc


def test_inference(model, vocab, model_name):
    """
    推理测试
    """
    print(f"\n--- {model_name} 推理示例 ---")
    model.eval()
    
    # 生成测试样本（明确 5 个字，确保"你"在正确位置）
    test_samples = [
        ("你好吗好的", 0),   # "你"在第 1 位
        ("我你好吗好", 1),   # "你"在第 2 位
        ("我你好的的", 1),   # "你"在第 2 位
        ("真的真的你", 4),   # "你"在第 5 位
        ("就是就是你", 4),   # "你"在第 5 位
    ]
    
    with torch.no_grad():
        for sent, true_label in test_samples:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            
            # 计算每个位置的概率
            probs = prob[0].tolist()
            prob_str = ", ".join([f"P{i+1}={p:.3f}" for i, p in enumerate(probs)])
            
            status = "✓" if pred == true_label else "✗"
            print(f"  {status} [{sent}] → 预测：第{pred+1}类，真实：第{true_label+1}类  [{prob_str}]")


def main():
    print("="*60)
    print("第三周作业：多分类任务 - 检测'你'字位置")
    print("="*60)
    
    # 1. 生成数据集
    print("\n生成数据集...")
    data = build_dataset(N_SAMPLES)
    
    # 2. 构建词表
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}")
    print(f"  词表大小：{len(vocab)}")
    
    # 3. 划分训练集和验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    print(f"  训练集：{len(train_data)}，验证集：{len(val_data)}")
    
    # 4. 创建 DataLoader
    train_loader = DataLoader(
        TextDataset(train_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val_data, vocab),
        batch_size=BATCH_SIZE
    )
    
    # 5. 训练不同模型
    models = [
        (RNNClassifier, "简单 RNN"),
        (LSTMClassifier, "双向 LSTM"),
        (GRUClassifier, "双向 GRU"),
        (AttentionLSTMClassifier, "Attention LSTM"),
    ]
    
    results = []
    
    for model_class, model_name in models:
        model, best_acc = train_model(
            model_class, model_name,
            train_loader, val_loader, vocab_size=len(vocab)
        )
        results.append((model_name, best_acc))
        
        # 推理测试
        test_inference(model, vocab, model_name)
    
    # 6. 汇总结果
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    print(f"{'模型':<20} {'验证准确率':<15}")
    print("-"*60)
    for model_name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{model_name:<20} {acc:.4f}")
    print("="*60)
    
    # 7. 额外测试：随机生成样本
    print("\n--- 随机样本测试 ---")
    print("使用双向 LSTM 模型进行 10 次随机测试：")
    
    lstm_model, _ = train_model(
        LSTMClassifier, "双向 LSTM（额外测试）",
        train_loader, val_loader, vocab_size=len(vocab)
    )
    
    correct = 0
    for i in range(10):
        true_pos = random.randint(1, 5)
        sent, label = generate_sample(true_pos)
        
        ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
        with torch.no_grad():
            logits = lstm_model(ids)
            pred = logits.argmax(dim=1).item()
        
        if pred == label:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} [{sent}] → 预测：第{pred+1}类，真实：第{label+1}类")
    
    print(f"\n随机测试准确率：{correct}/10 = {correct/10:.2%}")


if __name__ == '__main__':
    main()
