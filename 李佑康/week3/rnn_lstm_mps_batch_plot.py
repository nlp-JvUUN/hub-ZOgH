import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ======================
# 0. 设备（Mac GPU: MPS）
# ======================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ======================
# 1. 数据生成
# ======================

vocab = list("我他她它们是的了在看有不")
target_char = "你"

def generate_sample():
    pos = random.randint(0, 4)
    sentence = [random.choice(vocab) for _ in range(5)]
    sentence[pos] = target_char
    return "".join(sentence), pos

def generate_dataset(size=10000):
    return [generate_sample() for _ in range(size)]

data = generate_dataset(10000)

# ======================
# 2. 构建词表
# ======================

all_text = "".join([s for s, _ in data])
chars = list(set(all_text))
char2idx = {c: i for i, c in enumerate(chars)}

# ======================
# 3. 编码
# ======================

def encode(sentence):
    return [char2idx[c] for c in sentence]

X = torch.tensor([encode(s) for s, _ in data], dtype=torch.long)
y = torch.tensor([label for _, label in data], dtype=torch.long)

# 划分训练 / 测试
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ======================
# 4. DataLoader（mini-batch）
# ======================

from torch.utils.data import DataLoader, TensorDataset

batch_size = 64

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size
)

# ======================
# 5. 模型
# ======================

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ======================
# 6. 训练函数
# ======================

def train(model, train_loader, test_loader, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = outputs.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # 测试集
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                pred = outputs.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        test_acc = correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, TestAcc={test_acc:.4f}")

    return train_losses, train_accs, test_accs

# ======================
# 7. 训练两个模型
# ======================

vocab_size = len(char2idx)

print("\n====== RNN ======")
rnn_model = RNNModel(vocab_size)
rnn_loss, rnn_train_acc, rnn_test_acc = train(rnn_model, train_loader, test_loader)

print("\n====== LSTM ======")
lstm_model = LSTMModel(vocab_size)
lstm_loss, lstm_train_acc, lstm_test_acc = train(lstm_model, train_loader, test_loader)

# ======================
# 8. 画图
# ======================

plt.figure()

# Loss
plt.plot(rnn_loss, label="RNN Loss")
plt.plot(lstm_loss, label="LSTM Loss")

# Accuracy
plt.plot(rnn_train_acc, linestyle="--", label="RNN Train Acc")
plt.plot(lstm_train_acc, linestyle="--", label="LSTM Train Acc")
plt.plot(rnn_test_acc, linestyle=":", label="RNN Test Acc")
plt.plot(lstm_test_acc, linestyle=":", label="LSTM Test Acc")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("RNN vs LSTM")

plt.savefig("training_curve.png")
plt.show()

# ======================
# 9. 简单测试
# ======================

def predict(model, sentence):
    model.eval()
    x = torch.tensor([encode(sentence)], dtype=torch.long).to(device)
    out = model(x)
    return torch.argmax(out, dim=1).item()

print("\n====== 测试 ======")

tests = ["你我他她它", "我你他她它", "我他你她它"]

for s in tests:
    print(s, "->", predict(lstm_model, s))