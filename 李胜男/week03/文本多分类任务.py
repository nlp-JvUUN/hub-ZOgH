"""
文本多分类任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000
SEQ_LEN = 5  # 固定5个字
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字（约2000个常用字）
COMMON_CHARS = "的一是不了在人有我他这个中大来上们到说国和地也子时道出要就下以生会自作过动学对可主年发能工多同成行面所方后作部而分心样干都向力理体实家定深法水着机现所力起两长政现所意"


def generate_sample():
    """生成一个样本：5个字，必须包含'你'字"""
    # 随机选择'你'的位置
    target_pos = random.randint(0, SEQ_LEN - 1)

    # 生成文本
    chars = []
    for i in range(SEQ_LEN):
        if i == target_pos:
            chars.append('你')
        else:
            # 从常用字中随机选，但不能是'你'
            while True:
                ch = random.choice(COMMON_CHARS)
                if ch != '你':
                    chars.append(ch)
                    break

    text = ''.join(chars)
    return text, target_pos


def build_dataset(n=N_SAMPLES):
    """构建数据集"""
    data = []
    for _ in range(n):
        text, label = generate_sample()
        data.append((text, label))

    # 打乱
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """构建词汇表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)

    # 确保"你"在词表中
    if '你' not in vocab:
        vocab['你'] = len(vocab)

    return vocab


def encode(text, vocab, seq_len=SEQ_LEN):
    """将文本编码为ID序列"""
    ids = [vocab.get(ch, 1) for ch in text]  # 1是<UNK>
    ids = ids[:seq_len]
    ids += [0] * (seq_len - len(ids))  # 0是<PAD>
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class BaseRNNModel(nn.Module):
    """基础RNN模型（可替换为RNN/LSTM/GRU）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='lstm', num_classes=5, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 选择RNN类型
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        else:  # 默认RNN
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=False)

        self.rnn_type = rnn_type
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embeds = self.embedding(x)  # (batch, seq_len, embed_dim)

        # RNN处理
        if self.rnn_type == 'lstm':
            output, (hidden, _) = self.rnn(embeds)
        else:
            output, hidden = self.rnn(embeds)

        # 取最后一个时间步的隐藏状态
        if isinstance(hidden, tuple):  # LSTM返回的是(hidden, cell)
            hidden_state = hidden[0]
        else:
            hidden_state = hidden

        # hidden_state: (1, batch, hidden_dim) -> (batch, hidden_dim)
        last_hidden = hidden_state.squeeze(0)

        # 分类
        out = self.fc(self.dropout(last_hidden))
        return out


class MaxPoolRNNModel(nn.Module):
    """使用MaxPooling的RNN模型（对比用）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='lstm', num_classes=5, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=False)

        self.rnn_type = rnn_type
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)

        if self.rnn_type == 'lstm':
            output, _ = self.rnn(embeds)
        else:
            output, _ = self.rnn(embeds)

        # 最大池化
        pooled = output.max(dim=1)[0]  # (batch, hidden_dim)
        out = self.fc(self.dropout(pooled))
        return out


# ─── 5. 训练与评估函数 ──────────────────────────────────────────
def train_model(model, train_loader, val_loader, model_name="Model"):
    """训练一个模型并返回验证结果"""
    print(f"\n{'=' * 60}")
    print(f"训练模型: {model_name}")
    print('=' * 60)

    criterion = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 记录训练过程
    train_losses = []
    val_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        val_acc = evaluate_model(model, val_loader)
        val_accuracies.append(val_acc)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 最终评估
    final_acc = evaluate_model(model, val_loader, verbose=True)
    return final_acc, train_losses, val_accuracies


def evaluate_model(model, loader, verbose=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    if verbose:
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds,
                                    target_names=[f'位置{i}' for i in range(5)]))

    return acc


# ─── 6. 对比实验 ──────────────────────────────────────────
def main():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    print(f"样本数: {len(data)}")
    print(f"词表大小: {len(vocab)}")
    print(f"'你'字的ID: {vocab.get('你', '未找到')}")

    # 显示一些样本示例
    print("\n示例样本:")
    for i in range(5):
        text, label = data[i]
        print(f"  '{text}' → 位置{label}")

    # 划分训练集和验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")

    # 对比不同模型
    models_to_train = [
        ('RNN_last', BaseRNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 'rnn', num_classes=5)),
        ('LSTM_last', BaseRNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 'lstm', num_classes=5)),
        ('GRU_last', BaseRNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 'gru', num_classes=5)),
        ('RNN_maxpool', MaxPoolRNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 'rnn', num_classes=5)),
        ('LSTM_maxpool', MaxPoolRNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 'lstm', num_classes=5)),
    ]

    results = {}
    for model_name, model in models_to_train:
        print(f"\n初始化 {model_name}...")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")

        # 训练模型
        final_acc, train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, model_name
        )
        results[model_name] = {
            'acc': final_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

    # 对比结果
    print("\n" + "=" * 60)
    print("模型对比结果:")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"{model_name:15s} 最终验证准确率: {result['acc']:.4f}")

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['acc'])
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['acc']:.4f})")

    # 用最佳模型进行推理示例
    print("\n" + "=" * 60)
    print("推理示例:")
    print("=" * 60)

    # 使用第一个LSTM模型进行推理
    test_model = models_to_train[1][1]  # 取LSTM_last模型
    test_model.eval()

    test_samples = [
        "你好世界啊",  # 位置0
        "今天你开心",  # 位置2
        "欢迎你来到",  # 位置2
        "大家喜欢你",  # 位置3
        "我们支持你",  # 位置3
        "你真的很棒",  # 位置0
    ]

    with torch.no_grad():
        for text in test_samples:
            if len(text) != SEQ_LEN or '你' not in text:
                print(f"警告: '{text}' 不符合要求(必须5个字且包含'你')")
                continue

            # 找到真实位置
            true_label = text.index('你')

            # 编码
            ids = encode(text, vocab)
            x = torch.tensor([ids], dtype=torch.long)

            # 预测
            outputs = test_model(x)
            probs = torch.softmax(outputs, dim=1)
            pred_prob, pred_label = torch.max(probs, 1)

            pred_label = pred_label.item()
            pred_prob = pred_prob.item()

            correct = "✓" if pred_label == true_label else "✗"
            print(f"  {correct} '{text}' → 预测:位置{pred_label}({pred_prob:.2f}) 真实:位置{true_label}")


if __name__ == '__main__':
    main()
