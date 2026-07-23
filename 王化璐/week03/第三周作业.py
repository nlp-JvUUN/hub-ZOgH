import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# ===================== 1. 固定随机种子，保证实验可复现 =====================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ===================== 2. 任务超参数配置 =====================
SEQ_LEN = 5           # 固定输入文本长度：5个字
NUM_CLASSES = 5       # 分类数量：5类（对应「你」的5个位置）
EMBEDDING_DIM = 64    # 词嵌入维度
HIDDEN_DIM = 128      # RNN/LSTM隐藏层维度
BATCH_SIZE = 32       # 批次大小
EPOCHS = 5            # 训练轮数
LR = 1e-3             # 学习率

# 常用中文字符（用于生成非「你」的位置）
COMMON_CHARS = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "天", "地", "人", "风", "云", "雨", "雪", "日", "月", "星",
    "山", "水", "花", "鸟", "鱼", "虫", "我", "他", "她", "它"
]

# ===================== 3. 生成数据集 =====================
def generate_data(num_samples: int = 10000):
    """
    生成5字文本数据集：每个文本仅含1个「你」，位置随机
    返回：文本列表, 标签列表（标签=「你」的位置索引 0-4）
    """
    texts = []
    labels = []
    for _ in range(num_samples):
        # 随机选择「你」的位置 (0~4)
        target_pos = random.randint(0, SEQ_LEN - 1)
        # 生成5字文本
        text = []
        for i in range(SEQ_LEN):
            if i == target_pos:
                text.append("你")
            else:
                text.append(random.choice(COMMON_CHARS))
        texts.append("".join(text))
        labels.append(target_pos)
    return texts, labels

# 生成10000条训练数据，2000条测试数据
train_texts, train_labels = generate_data(10000)
test_texts, test_labels = generate_data(2000)

# ===================== 4. 文本预处理：汉字转数字 =====================
class Vocab:
    """构建汉字-索引映射词典"""
    def __init__(self, all_chars):
        self.char2idx = {"<PAD>": 0}  # 填充符（本任务固定长度，用不到）
        self.idx2char = {0: "<PAD>"}
        # 为所有字符分配唯一索引
        for char in all_chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = char
        self.vocab_size = len(self.char2idx)

# 收集所有字符，构建词典
all_chars = COMMON_CHARS + ["你"]
vocab = Vocab(all_chars)

def text2tensor(text: str):
    """将5字文本转换为张量（模型输入格式）"""
    indices = [vocab.char2idx[char] for char in text]
    return torch.tensor(indices, dtype=torch.long)

# ===================== 5. 自定义数据集 =====================
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [text2tensor(t) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 构建数据加载器
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 6. 定义RNN & LSTM 模型 =====================
# 基础RNN模型
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 词嵌入层：汉字索引 → 向量
        self.embedding = nn.Embedding(vocab.vocab_size, EMBEDDING_DIM)
        # RNN层
        self.rnn = nn.RNN(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        # 分类头：输出5分类
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        out, _ = self.rnn(x)   # out: [batch_size, seq_len, hidden_dim]
        # 取序列最后一个时间步的输出做分类
        out = out[:, -1, :]    # [batch_size, hidden_dim]
        return self.fc(out)    # [batch_size, num_classes]

# LSTM模型（解决RNN梯度消失问题，效果更好）
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, EMBEDDING_DIM)
        # LSTM层
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ===================== 7. 训练&评估函数 =====================
def train(model, train_loader, test_loader, epochs, model_name):
    # 设备：GPU优先，无GPU则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 损失函数：交叉熵（多分类任务标配）
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n===== 开始训练 {model_name} =====")
    for epoch in range(epochs):
        # 训练模式
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_text, batch_label in train_loader:
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            # 前向传播
            outputs = model(batch_text)
            loss = criterion(outputs, batch_label)
            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计准确率
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == batch_label).sum().item()
            total += batch_label.size(0)
        
        # 每轮打印训练结果
        train_acc = correct / total
        # 测试集评估
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")
    return model

def evaluate(model, test_loader, device):
    """评估模型在测试集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_text, batch_label in test_loader:
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            outputs = model(batch_text)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == batch_label).sum().item()
            total += batch_label.size(0)
    return correct / total

# ===================== 8. 预测函数（输入文本，输出「你」的位置） =====================
def predict_position(model, text: str):
    """
    输入：5个汉字的文本（必须含1个「你」）
    输出：「你」在文本中的位置（1~5）
    """
    if len(text) != SEQ_LEN:
        return "输入必须是5个汉字！"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tensor = text2tensor(text).unsqueeze(0).to(device)  # 增加batch维度
        output = model(tensor)
        pred_class = torch.argmax(output, dim=1).item()
        return f"文本：{text} | 「你」在第 {pred_class + 1} 位"

# ===================== 9. 主程序：训练+预测 =====================
if __name__ == "__main__":
    # 1. 训练RNN模型
    rnn_model = RNNModel()
    rnn_model = train(rnn_model, train_loader, test_loader, EPOCHS, "RNN")
    
    # 2. 训练LSTM模型
    lstm_model = LSTMModel()
    lstm_model = train(lstm_model, train_loader, test_loader, EPOCHS, "LSTM")
    
    # 3. 测试预测效果
    print("\n===== 模型预测示例 =====")
    test_samples = ["你一二三四", "一你二三三", "二一你四五", "三二四你一", "五四三二你"]
    for text in test_samples:
        print(predict_position(lstm_model, text))
