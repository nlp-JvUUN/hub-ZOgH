"""
中文五字文本位置分类作业

任务：
给定一个长度为 5 的中文文本，文本中一定包含一个“你”字。
判断“你”出现在第几位。

例子：
"你今天很好" -> 第 1 类，标签 0
"我你今天好" -> 第 2 类，标签 1
"我们你很好" -> 第 3 类，标签 2
"今天天你晴" -> 第 4 类，标签 3
"今天很好你" -> 第 5 类，标签 4

模型：
Embedding -> BiLSTM -> Dropout -> Linear

优化：
损失函数：CrossEntropyLoss
优化器：Adam
学习率：1e-3
"""


# ==================== 1. 导入需要的库 ====================
# 备注：
# os 用来设置环境变量。
# random 用来随机生成训练样本。
# torch 是 PyTorch 主库。
# torch.nn 里有神经网络层，比如 Embedding、LSTM、Linear。
# Dataset 和 DataLoader 用来组织数据和批量读取数据。
import os
import random

# 备注：
# 这行用于解决部分 Windows / Anaconda 环境中的 OpenMP 重复加载问题。
# 必须放在 import torch 之前。
# 如果你的环境不会报 libiomp5md.dll 错误，可以删除这一行。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ==================== 2. 设置超参数 ====================
# 备注：
# 超参数不是模型自动学出来的，而是我们提前设置好的训练配置。
SEED = 42              # 随机种子，让每次运行生成的数据尽量一致
N_SAMPLES = 10000      # 总样本数
SEQ_LEN = 5            # 每个文本固定 5 个字
EMBED_DIM = 64         # 每个字会被表示成 64 维向量
HIDDEN_DIM = 128       # LSTM 隐藏层维度
BATCH_SIZE = 64        # 每次训练取 64 条样本
EPOCHS = 10            # 整个训练集重复训练 10 轮
LR = 1e-3              # Adam 优化器的学习率
TRAIN_RATIO = 0.8      # 80% 数据训练，20% 数据验证


# 固定随机种子。
# 备注：
# 如果不固定，每次运行生成的数据和模型初始参数都会不同。
random.seed(SEED)
torch.manual_seed(SEED)


# ==================== 3. 准备用于随机填充的中文字符 ====================
# 备注：
# 这里不能包含“你”，否则一个样本里可能出现多个“你”，标签就不明确了。
# 后面生成样本时，会从这些字里随机选 4 个，再插入一个“你”。
FILLER_CHARS = list("我他她它们的是在有和了不很也都去来学习生活天气快乐朋友中国北京上海程序代码")


# ==================== 4. 生成单条样本 ====================
# 备注：
# 这个函数生成一个长度为 5 的文本。
# 文本里只有一个“你”。
# 返回值是：(文本, 标签)。
def make_sample():
    # 随机选择“你”的位置。
    # 位置范围是 0 到 4，对应第 1 位到第 5 位。
    ni_pos = random.randint(0, SEQ_LEN - 1)

    chars = []
    for i in range(SEQ_LEN):
        if i == ni_pos:
            chars.append("你")
        else:
            chars.append(random.choice(FILLER_CHARS))

    text = "".join(chars)

    # 标签直接使用 ni_pos。
    # 例如 ni_pos = 0 表示“你”在第 1 位，也就是标签 0。
    label = ni_pos
    return text, label


# ==================== 5. 构建完整数据集 ====================
# 备注：
# 数据集格式是列表：
# [
#     ("你今天很好", 0),
#     ("我你今天好", 1),
#     ...
# ]
def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_sample())

    # 打乱数据，避免样本顺序对训练造成影响。
    random.shuffle(data)
    return data


# ==================== 6. 构建字符级词表 ====================
# 备注：
# 神经网络不能直接处理汉字，需要先把每个汉字映射成数字 id。
# 例如：
# "我" -> 2
# "你" -> 3
# "好" -> 4
def build_vocab(data):
    # <PAD> 表示补齐字符，这个任务固定 5 字，暂时用不到，但保留是好习惯。
    # <UNK> 表示未知字符，如果预测时遇到词表外的字，就用它代替。
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)

    return vocab


# ==================== 7. 文本编码 ====================
# 备注：
# encode 的作用是把中文文本变成数字列表。
# 例如：
# "我你很好" -> [2, 3, 4, 5, 6]
def encode(text, vocab):
    ids = []
    for ch in text:
        ids.append(vocab.get(ch, vocab["<UNK>"]))
    return ids


# ==================== 8. 自定义 Dataset ====================
# 备注：
# PyTorch 的 DataLoader 需要配合 Dataset 使用。
# Dataset 负责告诉 PyTorch：
# - 一共有多少条数据
# - 按编号取某一条数据时，应该返回什么
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = []
        self.y = []

        for text, label in data:
            self.x.append(encode(text, vocab))
            self.y.append(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # 输入 x 必须是 long 类型，因为 Embedding 层接收的是整数 id。
        x = torch.tensor(self.x[index], dtype=torch.long)

        # 标签 y 也必须是 long 类型，因为 CrossEntropyLoss 要求类别编号。
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y


# ==================== 9. 定义模型 ====================
# 备注：
# 这是一个 5 分类模型。
# 输入 shape:  (batch_size, 5)
# 输出 shape:  (batch_size, 5)
# 输出的 5 个数分别表示属于 5 个类别的分数。
class NiPositionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding：
        # 把每个字符 id 转成一个向量。
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=0,
        )

        # BiLSTM：
        # bidirectional=True 表示双向 LSTM。
        # 它能同时利用左到右、右到左两个方向的信息。
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout：
        # 训练时随机丢弃一部分神经元，减少过拟合。
        self.dropout = nn.Dropout(0.3)

        # Linear：
        # BiLSTM 每个位置输出 HIDDEN_DIM * 2 维。
        # 一共有 5 个位置，所以展开后是 HIDDEN_DIM * 2 * 5 维。
        # 最后输出 5 个类别分数。
        self.fc = nn.Linear(HIDDEN_DIM * 2 * SEQ_LEN, SEQ_LEN)

    def forward(self, x):
        # x: (batch_size, 5)
        emb = self.embedding(x)
        # emb: (batch_size, 5, EMBED_DIM)

        lstm_out, _ = self.lstm(emb)
        # lstm_out: (batch_size, 5, HIDDEN_DIM * 2)

        # 把每个位置的 LSTM 输出拼接成一个长向量。
        features = lstm_out.reshape(lstm_out.size(0), -1)
        # features: (batch_size, HIDDEN_DIM * 2 * 5)

        features = self.dropout(features)

        logits = self.fc(features)
        # logits: (batch_size, 5)

        # 注意：
        # 这里不要加 softmax。
        # CrossEntropyLoss 会在内部自动计算 softmax。
        return logits


# ==================== 10. 验证函数 ====================
# 备注：
# 训练时要看模型在验证集上的准确率，避免只看 loss。
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    # 验证阶段不需要计算梯度，可以节省内存和时间。
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)

            # argmax(dim=1) 表示在 5 个类别分数中取最大值所在的位置。
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ==================== 11. 训练函数 ====================
def train():
    # 1. 生成数据。
    data = build_dataset(N_SAMPLES)

    # 2. 构建词表。
    vocab = build_vocab(data)

    # 3. 划分训练集和验证集。
    split_index = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # 4. 创建 DataLoader。
    train_loader = DataLoader(
        PositionDataset(train_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        PositionDataset(val_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 5. 创建模型。
    model = NiPositionModel(vocab_size=len(vocab))

    # 6. 多分类任务使用 CrossEntropyLoss。
    criterion = nn.CrossEntropyLoss()

    # 7. 使用 Adam 优化器更新模型参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("任务：判断五字文本中“你”的位置")
    print("模型：Embedding + BiLSTM + Dropout + Linear")
    print("优化：Adam + CrossEntropyLoss")
    print(f"样本数：{len(data)}")
    print(f"词表大小：{len(vocab)}")
    print()

    # 8. 开始训练。
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            # 前向传播：得到每个类别的分数。
            logits = model(x)

            # 计算损失。
            loss = criterion(logits, y)

            # 清空上一轮的梯度。
            optimizer.zero_grad()

            # 反向传播：计算梯度。
            loss.backward()

            # 更新参数。
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 训练结束后，跑几个示例看看效果。
    predict_examples(model, vocab)


# ==================== 12. 预测示例 ====================
def predict_examples(model, vocab):
    examples = [
        "你今天很好",
        "我你今天好",
        "我们你很好",
        "今天天你晴",
        "今天很好你",
    ]

    print()
    print("预测示例：")

    model.eval()
    with torch.no_grad():
        for text in examples:
            x = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            print(f"{text} -> 你在第 {pred + 1} 位")


# ==================== 13. 程序入口 ====================
# 备注：
# 只有直接运行这个文件时，才会执行 train()。
# 如果这个文件被别的 Python 文件 import，就不会自动训练。
if __name__ == "__main__":
    train()
