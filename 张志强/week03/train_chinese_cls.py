"""
train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本 使用交叉熵分类练习

任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
模型：Embedding → RNN → 取最后隐藏状态 → Linear
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss 多分类交叉熵   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42  # 随机种子，确保实验结果可复现
N_SAMPLES = 4000  # 训练样本数量
MAXLEN = 5  # 文本序列最大长度（词数）
EMBED_DIM = 64  # 词嵌入向量维度
HIDDEN_DIM = 64  # RNN隐藏层维度
LR = 1e-3  # 学习率
BATCH_SIZE = 64  # 批量大小
EPOCHS = 20  # 训练轮数
TRAIN_RATIO = 0.8  # 训练集占比（剩余20%用于验证）


random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
KEY_CHAR = "你"
WORDS_LIST = list("的一是不了人我有中大家这为都以到起把看和去会能自说的上好过小么要下可出也")

#随机生成一个带关键字的五个字的句子
def make_keywords():
    # 随机选择“你”的位置 (0~4)
    pos = random.randint(0, 4)
    chars = [random.choice(WORDS_LIST) for _ in range(MAXLEN)]
    # 在指定位置放“你”（确保只有这一个“你”）
    chars[pos] = KEY_CHAR
    text = "".join(chars)
    return text, pos

# 构建出N_SAMPLES条数据  里面包含一半的正样本  一半的负样本
def build_dataset(n=N_SAMPLES):
    data = []
    # 获取到4000//2=2000的数据   make_positive()是正样本 标志成1    make_negative负样本 标志成0
    for _ in range(n):
        words_str, pos = make_keywords()
        data.append((words_str, pos))
    # 打乱顺序
    random.shuffle(data)
    print(f"生成数据条数: {len(data)}，标签示例: {data[:3]}")
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    # 构建词表  <PAD> 是为了句子保持一样长度  不够的则用<PAD>填充 <UNK>是未匹配到的
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            # 如果遍历的某个字没在vocab词表里 则加入
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


# 获取每个词对应的id  如果词在词表vocab没找到id就为1   数据不够填充0  然后最大长度是maxlen
def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ss = [0] * (maxlen - len(ids))
    # [0] * (maxlen - len(ids)) 生成一个长度为maxlen - len(ids)的列表，列表中的元素都是0。
    # 用法相当于 [0] *2  print("=="*20) 并拼接到ids的后边  为了等长MAXLEN
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    #data 训练的数据集  vocab:对应的词表
    def __init__(self, data, vocab):
        #x是得到句子中的词 对应的id是什么  比如 "今天天气阴沉"  今 3 天 4 气 5
        self.X = [encode(s, vocab) for s, _ in data]
        #y是存储的句子的正负样本的数据  1, 1, 0, 1, 0, 1, 0, 0, 0, 0 等
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # 将索引编码和标签转换为 PyTorch 张量，分别指定长整型和浮点型以适配模型输入与损失函数计算
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            #真实标签 y：形状为 (B,) 的 长整型 (torch.long)  需要改为long
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        # 词嵌入embedding 将词转化为向量  embed_dim:向量维度  padding_idx=0:PAD不更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN 卷积层 提取序列特征
        # batch_first=True:规定输入/输出张量的维度顺序把 batch 放在第 1 维。 设置后输入形状是 (B, L, input_size)，输出形状是 (B, L, hidden_size)。不设置（默认 False）时形状会变成 (L, B, input_size) / (L, B, hidden_size)。
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        #批归一化层：对 RNN 输出的特征进行标准化，加速收敛并提高训练稳定性    为了将每层输入拉会均值0,方差1附近
        self.bn = nn.BatchNorm1d(hidden_dim)
        # 丢弃层：随机丢弃一部分神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 全连接层(线性层)：将 RNN 输出映射到标签空间  输出维度改为类别数5（位置0~4）
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        #这段代码执行的是 Max Pooling（最大池化） 操作，目的是从变长的序列中提取出最重要的特征。
        #理解为把句子“压扁”。不管句子原来有 10 个字还是 100 个字，通过在这个方向(纵向)上取最大值，我们只保留了整个句子中最“突出”的那些特征信号，
        #PyTorch 的 .max() 函数会返回一个包含两个元素的元组：(最大值, 最大值的索引) [0] 表示只取元组中的第一个元素，也就是我们需要的最大值张量。
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        # self.bn 该代码对张量pooled依次执行批归一化 然后再Dropout正则化
        pooled = self.dropout(self.bn(pooled))
        #经过全连接层   (B,5)
        out = self.fc(pooled)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob = model(X)  # (B, 5)
            pred = prob.argmax(dim=1)  # (B,)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    print("生成数据集...")
    #N_SAMPLES=4000  生成4000千条数据 数据里存储正样本的文案和标志
    data = build_dataset(N_SAMPLES)
    #构建词表 词表就是字典集合里 字对应的的id eg: "今天天气阴沉"  今:3 天:4 气:5
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    #取数据的80%用于训练  训练集合
    train_data = data[:split]
    #取数据的20%用于验证  验证集合
    val_data = data[split:]

    #shuffle=True 打乱顺序
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    #多分类 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器  用于计算权重梯度
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #p.numel()：返回这个参数张量里元素的总个数（number of elements），也就是这个张量包含多少个可学习数值  sum(...)：把所有参数张量的元素数加起来，得到整个模型的总参数量。
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            #预测值和真实值 计算损失
            loss = criterion(pred, y)
            # 梯度归零
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        "你好世界啊",  # 位置0
        "你是谁啊",  # 位置1
        "你说点好话吧",  # 位置1
        "我很想你啊",  # 位置3
        "今天吃你饭",  # 位置4
        "你我他大家",  # 位置0
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob = model(ids)
            #找到预测值的最大索引
            pred_pos = prob.argmax(dim=1).item()
            true_pos = sent.find(KEY_CHAR)
            print(f"  输入: {sent} | 预测位置: {pred_pos} (第{pred_pos}位) | 实际位置: {true_pos}")


if __name__ == '__main__':
    train()
