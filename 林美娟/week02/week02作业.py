# 作业：完成一个多分类任务的训练，一个随机向量，哪一维数字最大就属于第几类
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 超参数设置 =====================
INPUT_DIM = 10   # 输入随机向量的维度（可修改）
NUM_CLASSES = 16  # 分类数 = 向量维度（最大值索引就是类别）
BATCH_SIZE = 32   # 批次大小
EPOCHS = 50       # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 2. 数据生成函数 =====================
def generate_random_data(batch_size, input_dim):
    """
    生成随机向量数据 + 标签
    标签规则：向量中最大值所在的索引 = 分类标签
    """
    # 生成标准正态分布随机向量
    data = torch.randn(batch_size, input_dim).to(DEVICE)
    # 计算最大值索引 → 标签
    labels = torch.argmax(data, dim=1).to(DEVICE)
    return data, labels

# ===================== 3. 构建分类模型 =====================
class VectorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VectorClassifier, self).__init__()
        # 全连接网络
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 输出：每个类别的预测分数
        )

    def forward(self, x):
        return self.fc(x)

# ===================== 4. 初始化模型、损失、优化器 =====================
model = VectorClassifier(INPUT_DIM, NUM_CLASSES).to(DEVICE)
# 多分类任务标准损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================== 5. 开始训练 =====================
print(f"训练设备: {DEVICE}\n开始训练...\n")
model.train()  # 开启训练模式
for epoch in range(EPOCHS):
    # 生成一批随机训练数据
    data, labels = generate_random_data(BATCH_SIZE, INPUT_DIM)
    
    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, labels)
    
    # 反向传播 + 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    pred_labels = torch.argmax(outputs, dim=1)
    accuracy = (pred_labels == labels).float().mean()
    
    # 打印日志
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}")

# ===================== 6. 模型测试（推理验证） =====================
print("\n========== 测试模型 ==========")
model.eval()  # 开启评估模式
with torch.no_grad():
    # 生成1个测试随机向量
    test_vec, test_label = generate_random_data(1, INPUT_DIM)
    # 模型预测
    test_output = model(test_vec)
    test_pred = torch.argmax(test_output, dim=1).item()

# 打印结果
print(f"测试随机向量:\n{test_vec.cpu().numpy().round(3)}")
print(f"真实标签(最大值索引): {test_label.item()}")
print(f"模型预测标签: {test_pred}")
print(f"预测结果: {'正确' if test_pred == test_label.item() else '错误'}")
