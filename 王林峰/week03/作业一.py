# ===================== 导入依赖库 =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter

# ===================== 1. 固定随机种子（保证实验可复现） =====================
def set_seed(seed=42):
    """
    功能：固定所有随机数种子，让每次运行结果都一样
    参数：
        seed: 随机种子数，默认42（通用实验种子）
    举例：seed=42 → 所有随机操作都按固定顺序生成
    """
    random.seed(seed)       # Python基础随机数种子
    np.random.seed(seed)    # numpy随机数种子
    torch.manual_seed(seed) # torch CPU种子
    torch.cuda.manual_seed_all(seed) # torch GPU种子

# 执行固定种子
set_seed(seed=42)

# ===================== 2. 生成模拟多分类文本数据集（5分类） =====================
def generate_sample_data(num_samples=5000):
    """
    功能：自动生成5分类文本数据（不用自己准备数据集）
    参数：
        num_samples: 总数据量，默认5000条
                    举例：5000条 → 每类1000条（5类均等）
    返回：
        texts: 文本列表，如 ["AI robot data cloud", "football match win goal"]
        labels: 标签列表（0-4），如 [0,1]
    分类说明：
        0:科技  1:体育  2:娱乐  3:财经  4:教育
    """
    # 5个分类名称
    categories = ["tech", "sports", "entertainment", "finance", "education"]
    
    # 每个分类的专属词汇（模型靠这些词区分类别）
    vocab = {
        "tech": ["algorithm", "AI", "data", "computer", "robot", "cloud", "code", "server"],
        "sports": ["football", "basketball", "match", "athlete", "Olympic", "win", "goal"],
        "entertainment": ["movie", "music", "star", "concert", "film", "actor", "song"],
        "finance": ["stock", "market", "investment", "bank", "profit", "fund", "trade"],
        "education": ["school", "student", "teacher", "exam", "college", "study", "learn"]
    }

    texts = []   # 存储所有文本
    labels = []  # 存储所有标签

    # 遍历5个分类，逐个生成数据
    for idx, cat in enumerate(categories):
        words = vocab[cat]          # 当前分类的词汇表
        sample_num = num_samples // 5  # 每类样本数 = 总样本/5
        for _ in range(sample_num):
            sent_len = random.randint(5, 15)  # 句子长度：随机5~15个词
            sent = " ".join(random.choices(words, k=sent_len)) # 随机拼句子
            texts.append(sent)
            labels.append(idx)  # 标签：0,1,2,3,4
    
    # 打乱数据顺序（防止同类排在一起）
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts[:], labels[:] = zip(*combined)
    
    return texts, labels

# 生成数据集：5000条文本 + 5分类标签
texts, labels = generate_sample_data(num_samples=5000)

# 划分训练集(80%)、测试集(20%)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,          # 所有文本
    labels,         # 所有标签
    test_size=0.2,  # 测试集占比20%
    random_state=42 # 随机种子
)
"""
举例：
总数据5000 → 训练集4000条，测试集1000条
train_texts[0] = "AI robot data cloud code"
train_labels[0] = 0（科技类）
"""

# ===================== 3. 文本预处理：单词 → 数字编码 =====================
class TextProcessor:
    """
    文本处理器：把英文句子转成模型能看懂的数字序列
    流程：分词 → 构建词典 → 编码 → 统一长度(padding)
    """
    def __init__(self, max_vocab_size=500, max_seq_len=20):
        """
        初始化参数
        参数：
            max_vocab_size: 词典最大词量，默认500
                            举例：只保留频率最高的500个词
            max_seq_len: 句子最大长度，默认20
                         举例：超过20词截断，不足补0
        内置词典：
            <PAD>: 填充符 → 编号0
            <UNK>: 未知词 → 编号1
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1} # 词→数字
        self.idx2word = {0: "<PAD>", 1: "<UNK>"} # 数字→词

    def build_vocab(self, texts):
        """
        功能：从训练文本中统计词频，构建词典
        参数：
            texts: 训练集文本列表
        举例：
            词"AI"出现1000次 → 编入词典
            生僻词只出现1次 → 忽略
        """
        all_words = []
        for text in texts:
            all_words.extend(text.split()) # 拆分所有单词
        
        # 统计词频，取前max_vocab_size-2个高频词
        word_count = Counter(all_words)
        vocab = word_count.most_common(self.max_vocab_size - 2)
        
        # 给高频词分配编号（从2开始）
        for i, (word, _) in enumerate(vocab):
            self.word2idx[word] = i + 2
            self.idx2word[i + 2] = word

    def encode(self, text):
        """
        功能：把句子 → 数字序列
        参数：
            text: 单条文本，如 "AI robot data"
        返回：
            固定长度数字序列，如 [6,9,12,0,0,...0]
        """
        words = text.split() # 拆分句子
        # 词→编号，不在词典里用1（UNK）
        indices = [self.word2idx.get(word, 1) for word in words]
        
        # 统一长度：不足补0，超过截断
        if len(indices) < self.max_seq_len:
            indices += [0] * (self.max_seq_len - len(indices))
        else:
            indices = indices[:self.max_seq_len]
        
        return indices

# 初始化文本处理器
processor = TextProcessor(max_vocab_size=500, max_seq_len=20)
processor.build_vocab(train_texts) # 用训练集构建词典

# 把训练/测试文本编码成数字序列
train_encoded = [processor.encode(t) for t in train_texts]
test_encoded = [processor.encode(t) for t in test_texts]
"""
编码举例：
原句："AI robot data cloud"
编码后：[6,9,12,15,0,0,0,...0]（长度20）
"""

# 转成PyTorch张量（模型只能用tensor训练）
train_x = torch.tensor(train_encoded, dtype=torch.long)
test_x = torch.tensor(test_encoded, dtype=torch.long)
train_y = torch.tensor(train_labels, dtype=torch.long)
test_y = torch.tensor(test_labels, dtype=torch.long)

# ===================== 4. 构建数据集加载器 =====================
class TextDataset(Dataset):
    """
    PyTorch标准数据集格式
    功能：包装文本和标签，方便批量加载
    """
    def __init__(self, x, y):
        self.x = x # 编码后的数字序列
        self.y = y # 标签

    def __len__(self):
        return len(self.x) # 返回总样本数

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] # 按索引取单条数据

# 批次大小：每次喂给模型32条数据
batch_size = 32

# 训练加载器：打乱顺序 + 分批
train_loader = DataLoader(TextDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
# 测试加载器：不打乱 + 分批
test_loader = DataLoader(TextDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

# ===================== 5. 模型定义：RNN / LSTM / GRU =====================
# 1. RNN模型
class RNNClassifier(nn.Module):
    """
    RNN文本分类模型
    结构：Embedding → RNN → 全连接层 → 分类输出
    """
    def __init__(
        self,
        vocab_size,    # 词典大小，例：500
        embed_dim,     # 词向量维度，例：64
        hidden_dim,    # RNN隐层维度，例：128
        num_classes,   # 分类数，例：5
        num_layers=1   # RNN层数，默认1
    ):
        super().__init__()
        # 词嵌入层：数字 → 向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN层
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        # 分类全连接层：隐层 → 分类结果
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """前向传播：模型计算流程"""
        x = self.embedding(x)  # 输入[32,20] → 输出[32,20,64]（批次,序列长,词向量）
        out, _ = self.rnn(x)   # RNN输出[32,20,128]
        out = out[:, -1, :]    # 取最后一个时间步输出[32,128]
        out = self.fc(out)     # 输出分类[32,5]
        return out

# 2. LSTM模型（带门控，解决长序列依赖）
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# 3. GRU模型（LSTM简化版，速度更快）
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ===================== 6. 训练 + 评估函数 =====================
def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    """
    完整训练流程
    参数：
        model: RNN/LSTM/GRU模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数，例：8轮
        lr: 学习率，例：0.001
    """
    # 自动选择GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # 模型搬到设备上
    
    # 损失函数：多分类交叉熵
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam（更新模型参数）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 逐轮训练
    for epoch in range(epochs):
        # ========== 训练模式 ==========
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # 数据搬到GPU/CPU
            optimizer.zero_grad()             # 清空梯度
            outputs = model(x)                # 前向传播
            loss = criterion(outputs, y)      # 计算损失
            loss.backward()                   # 反向传播（算梯度）
            optimizer.step()                  # 更新参数

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)   # 取预测概率最大的类别
            correct += (pred == y).sum().item() # 统计正确数
            total += y.size(0)                # 总样本数

        train_acc = correct / total # 训练准确率

        # ========== 评估模式 ==========
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): # 不计算梯度，加速+省显存
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = correct / total # 测试准确率

        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    
    return model

# ===================== 7. 超参数设置 =====================
vocab_size = len(processor.word2idx) # 词典实际大小：约500
embed_dim = 64       # 词向量维度：每个词用64维向量表示
hidden_dim = 128     # RNN隐层神经元数：128
num_classes = 5      # 分类数：5类
epochs = 8           # 训练8轮
lr = 1e-3            # 学习率：0.001

# ===================== 8. 开始训练三个模型 =====================
print("="*50)
print("训练 RNN 模型")
rnn_model = RNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
train_model(rnn_model, train_loader, test_loader, epochs, lr)

print("\n" + "="*50)
print("训练 LSTM 模型")
lstm_model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
train_model(lstm_model, train_loader, test_loader, epochs, lr)

print("\n" + "="*50)
print("训练 GRU 模型")
gru_model = GRUClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
train_model(gru_model, train_loader, test_loader, epochs, lr)