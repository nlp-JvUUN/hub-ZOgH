import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random


# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42 #随机种子
LR          = 1e-3 #学习率
def generate_sample():
    """生成一个五个字的样本，"你"随机出现在1-5位"""
    words = ['九', '月', '日', '心', '天', '时', '地', '利', '人', '和', '爱', '我', '吗', '十', '们']
    pos = random.randint(0, 4)  # "你"出现的位置
    # 生成除了第pos位以外的其他字
    text = []
    for i in range(5):
        if i == pos:
            text.append('你')
        else:
            text.append(random.choice(words))
    return ''.join(text), pos


def build_dataset(n_samples=1000):
    """生成数据集"""
    data = []
    for _ in range(n_samples):
        text, label = generate_sample()
        data.append((text, label))
    return data


def text_to_tensor(text, char_to_idx):
    """将文本转换为张量"""
    return torch.tensor([char_to_idx.get(c, char_to_idx['<UNK>']) for c in text], dtype=torch.long)


class TextDataset(Dataset):
    """文本数据设置"""
    def __init__(self, data, char_to_idx):
        self.data = data
        self.char_to_idx = char_to_idx
        self.vocab_size = len(char_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        x = text_to_tensor(text, self.char_to_idx)
        return x, torch.tensor(label, dtype=torch.long)


class RNN(nn.Module):
    """rnn模型"""
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(RNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """前向传播"""
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class LSTM(nn.Module):
    """LSTM模型"""
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTM, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """前向传播"""
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class GRU(nn.Module):
    """GRU模型"""
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, dropout=0.3):
        super(GRU, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """前向传播"""
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def train_model(model, train_loader, optimizer, criterion):
    """训练模型"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {batch_idx} Loss: {loss.item():.6f}')
    return model


def test_model(model, test_loader, criterion):
    """测试模型"""
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
    print(f'Test Loss: {test_loss:.6f}, Accuracy: {correct / len(test_loader.dataset):.6f}')
    return model


def build_vocab(data):
    """构建字符到索引的映射"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def main():
    # 生成数据集
    n_samples = 2000
    train_samples = 1600
    data = build_dataset(n_samples)
    # 划分训练集和测试集
    train_data = data[:train_samples]
    test_data = data[train_samples:]
    # 构建词表
    vocab = build_vocab(data)
    # 创建数据加载器
    train_dataset = TextDataset(train_data, vocab)
    test_dataset = TextDataset(test_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # 训练 RNN 模型
    print("训练 RNN 模型...")
    vocab_size = len(vocab)
    embed_dim = 32
    hidden_size = 32
    model = RNN(vocab_size, embed_dim, hidden_size, num_layers=2, num_classes=5)
    optimizer = optim.Adam(model.parameters(), LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        train_model(model, train_loader, optimizer, criterion)
        test_model(model, test_loader, criterion)
    # 训练 LSTM 模型
    print("\n训练 LSTM 模型...")
    model = LSTM(vocab_size, embed_dim, hidden_size, num_layers=2, num_classes=5)
    optimizer = optim.Adam(model.parameters(), LR)
    for epoch in range(10):
        train_model(model, train_loader, optimizer, criterion)
        test_model(model, test_loader, criterion)
    # 训练 GRU 模型
    print("\n训练 GRU 模型...")
    model = GRU(vocab_size, embed_dim, hidden_size, num_layers=2, num_classes=5)
    optimizer = optim.Adam(model.parameters(), LR)
    for epoch in range(10):
        train_model(model, train_loader, optimizer, criterion)
        test_model(model, test_loader, criterion)
    print("\n所有模型训练完成！")
    # 测试示例
    print("\n--- 测试示例 ---")
    model.eval()
    test_texts = ["你是谁是额", "我四你蝶的", "全部都是你", "我不喜欢你", "一切都怪你","非你不可吗","九月是你了"]
    with torch.no_grad():
        for text in test_texts:
            x = text_to_tensor(text, vocab).unsqueeze(0)
            output = model(x)
            pred = output.argmax(dim=1).item()
            print(f"文本: '{text}' -> 预测位置: {pred + 1} (共5位)")


if __name__ == '__main__':
    main()
