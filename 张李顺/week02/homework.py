"""
多分类任务:输入一个随机向量,预测哪一维数字最大
例如: [0.1, 0.8, 0.3, 0.2] -> 类别 1 (第1维 0.8 最大)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ================================================
DIM = 5              # 向量维度(类别数)
NUM_SAMPLES = 100    # 总数据量
BATCH_SIZE = 16      # 批大小
EPOCHS = 50          # 训练轮数
LR = 0.01            # 学习率
TRAIN_RATIO = 0.8    # 训练集比例
SEED = 42            # 随机种子
# ================================================

torch.manual_seed(SEED)

# 1. 生成数据
# X: 随机向量, Y: 每行的argmax作为类别标签
X = torch.randn(NUM_SAMPLES, DIM)
Y = torch.argmax(X, dim=1)  # 标签就是最大值的索引

print(f"数据形状: X={X.shape}, Y={Y.shape}")
print(f"标签分布: {torch.bincount(Y, minlength=DIM).tolist()}")

# 2. 划分训练集和测试集 (8:2)
dataset = TensorDataset(X, Y)
train_size = int(NUM_SAMPLES * TRAIN_RATIO)
test_size = NUM_SAMPLES - train_size
train_set, test_set = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {train_size}, 测试集: {test_size}\n")


# 3. 定义一个简单的 MLP 模型
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x


model = Classifier(DIM, DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# 4. 评估函数
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


# 5. 训练
print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # 每 5 轮打印一次
    if epoch % 5 == 0 or epoch == 1:
        train_loss, train_acc = evaluate(train_loader)
        test_loss, test_acc = evaluate(test_loader)
        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

# 6. 最终测试 + 看几个样例
print("\n===== 最终结果 =====")
test_loss, test_acc = evaluate(test_loader)
print(f"测试集准确率: {test_acc:.3f}")

print("\n===== 几个预测样例 =====")
model.eval()
with torch.no_grad():
    sample_x, sample_y = next(iter(test_loader))
    pred = model(sample_x).argmax(dim=1)
    for i in range(min(5, len(sample_x))):
        vec = sample_x[i].tolist()
        vec_str = "[" + ", ".join(f"{v:+.2f}" for v in vec) + "]"
        ok = "✓" if pred[i] == sample_y[i] else "✗"
        print(f"{vec_str}  真实={sample_y[i].item()}  预测={pred[i].item()}  {ok}")
