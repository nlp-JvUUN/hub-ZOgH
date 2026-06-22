import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt


# 设置随机种子，确保实验的可重复性
random.seed(42)
torch.manual_seed(42)


# 生成自定义的包含“你”字的五个字文本样本及其对应的类别
def generate_custom_samples():
    samples = [
        ("你好呀哈哈", 0),
        ("我喜欢你呀", 2),
        ("他你在干嘛", 1),
        ("好呀你真棒", 2),
        ("呀你真好看", 1),
        ("真好你真行", 2),
        ("我你一起走", 1)
    ]
    return samples


# 构建数据集，这里直接返回生成的样本
def build_dataset(samples):
    return samples


# 将文本转换为张量，通过字符到索引的映射表
def text_to_tensor(text, char_to_idx):
    return torch.tensor([char_to_idx.get(c, char_to_idx['<UNK>']) for c in text], dtype=torch.long)


# 自定义数据集类，继承自Dataset
class TextDataset(Dataset):
    def __init__(self, data, char_to_idx):
        # 存储数据
        self.data = data
        # 存储字符到索引的映射表
        self.char_to_idx = char_to_idx
        # 词汇表大小
        self.vocab_size = len(char_to_idx)

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 获取数据集中指定索引位置的样本
    def __getitem__(self, idx):
        text, label = self.data[idx]
        x = text_to_tensor(text, self.char_to_idx)
        return x, torch.tensor(label, dtype=torch.long)


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(RNN, self).__init__()
        # 嵌入维度
        self.embed_dim = embed_dim
        # 隐藏层维度
        self.hidden_size = hidden_size
        # RNN层数
        self.num_layers = num_layers
        # 嵌入层，将字符索引转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN层
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 全连接层，将隐藏层输出映射到类别空间
        self.fc = nn.Linear(hidden_size, num_classes)

    # 前向传播
    def forward(self, x):
        x = self.embedding(x)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的隐藏状态
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTM, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM层
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(GRU, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # GRU层
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# 训练模型函数
def train_model(model, train_loader, optimizer, criterion, epochs):
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}')
    return train_losses


# 测试模型函数
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.6f}')
    return test_loss, accuracy


# 构建字符到索引的映射表
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


# 预测并可视化结果函数
def predict_and_visualize(model, test_texts, char_to_idx, model_name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in test_texts:
            x = text_to_tensor(text, char_to_idx).unsqueeze(0)
            output = model(x)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(test_texts)), predictions)
    plt.xticks(range(len(test_texts)), test_texts, rotation=45)
    plt.xlabel('Test Texts')
    plt.ylabel('Predicted Class')
    plt.title(f'Predictions using {model_name}')
    plt.show()


def main():
    custom_samples = generate_custom_samples()
    data = build_dataset(custom_samples)

    # 划分训练集和测试集，80%作为训练集，20%作为测试集
    train_samples = int(len(data) * 0.8)
    train_data = data[:train_samples]
    test_data = data[train_samples:]

    # 构建词汇表
    vocab = build_vocab(data)

    # 创建训练集和测试集的数据加载器
    train_dataset = TextDataset(train_data, vocab)
    test_dataset = TextDataset(test_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    models = {
        'RNN': RNN(len(vocab), 32, 32, 2, 5),
        'LSTM': LSTM(len(vocab), 32, 32, 2, 5),
        'GRU': GRU(len(vocab), 32, 32, 2, 5)
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        train_losses = train_model(model, train_loader, optimizer, criterion, epochs=50)
        test_loss, accuracy = test_model(model, test_loader, criterion)

        test_texts = ["你真好呀哈", "我好喜欢你", "他在你身边"]
        predict_and_visualize(model, test_texts, vocab, model_name)


if __name__ == '__main__':
    main()
