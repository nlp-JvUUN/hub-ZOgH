import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

# -------------------- 1. 生成数据集 --------------------
class YouPositionDataset(Dataset):
    def __init__(self, num_samples, char_set, seq_len=5):
        self.seq_len = seq_len
        self.char_set = char_set
        self.data = []
        self.labels = []
        
        # 生成样本
        for _ in range(num_samples):
            # 随机选择"你"的位置 (0~4)
            pos = random.randint(0, 4)
            # 生成其它位置的随机字符（不能是"你"）
            chars = []
            for i in range(seq_len):
                if i == pos:
                    chars.append('你')
                else:
                    # 从字符集中排除"你"
                    other_chars = [c for c in char_set if c != '你']
                    chars.append(random.choice(other_chars))
            self.data.append(''.join(chars))
            self.labels.append(pos)  # 0~4，对应类别1~5
        
        # 构建字符到索引的映射
        self.char2idx = {c: i for i, c in enumerate(char_set)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(char_set)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        # 将每个字符转为索引
        indices = [self.char2idx[ch] for ch in text]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 构建字符集（常用中文字符，保证包含"你"）
char_set = list('你我他她它的一是不了在有人和这中大来上国为个学小要时地出就年')
# 确保"你"在字符集中
if '你' not in char_set:
    char_set.append('你')
char_set = sorted(list(set(char_set)))  # 去重并排序

# 生成训练集和测试集
train_dataset = YouPositionDataset(5000, char_set)
test_dataset = YouPositionDataset(1000, char_set)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

vocab_size = train_dataset.vocab_size
print(f"词汇表大小: {vocab_size}")
print(f"训练样本数: {len(train_dataset)}")
print(f"测试样本数: {len(test_dataset)}")

# -------------------- 2. 定义模型 --------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)  # hidden: (num_layers, batch, hidden_dim)
        # 取最后一层的最后一个时刻的隐藏状态
        out = hidden[-1, :, :]  # (batch, hidden_dim)
        out = self.fc(out)      # (batch, output_dim)
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # 取最后一层最后一个时刻的隐藏状态
        out = hidden[-1, :, :]  # (batch, hidden_dim)
        out = self.fc(out)
        return out

# -------------------- 3. 训练函数 --------------------
def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_accs.append(acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")
    
    return train_losses, test_accs

# -------------------- 4. 运行实验 --------------------
# 超参数
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 5  # 5个类别
NUM_LAYERS = 1

# 创建两个模型
rnn_model = RNNClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
lstm_model = LSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)

print("训练 RNN ...")
rnn_losses, rnn_accs = train_model(rnn_model, train_loader, test_loader, epochs=20)

print("\n训练 LSTM ...")
lstm_losses, lstm_accs = train_model(lstm_model, train_loader, test_loader, epochs=20)

# -------------------- 5. 可视化对比 --------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(rnn_losses)+1), rnn_losses, label='RNN', marker='o')
plt.plot(range(1, len(lstm_losses)+1), lstm_losses, label='LSTM', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Loss Comparison')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(rnn_accs)+1), rnn_accs, label='RNN', marker='o')
plt.plot(range(1, len(lstm_accs)+1), lstm_accs, label='LSTM', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Accuracy Comparison')

plt.tight_layout()
plt.show()

# -------------------- 6. 推理示例 --------------------
def predict(model, text, dataset):
    model.eval()
    device = next(model.parameters()).device
    indices = [dataset.char2idx[ch] for ch in text]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item() + 1  # 转为1~5
    return pred

# 测试几个例子
test_texts = ['你喜欢我吗', '我你喜欢吗', '我你喜吗欢', '我喜欢你吗', '我喜欢吗你']
print("\n推理示例:")
for text in test_texts:
    pred_rnn = predict(rnn_model, text, train_dataset)
    pred_lstm = predict(lstm_model, text, train_dataset)
    print(f"'{text}' -> RNN预测: 第{pred_rnn}位, LSTM预测: 第{pred_lstm}位")