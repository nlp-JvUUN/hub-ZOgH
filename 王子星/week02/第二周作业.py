import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# 定义分类任务的神经网络
class TorchModle(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.fuc1 = nn.Linear(input_size, hidden_size)
        # 隐藏层的激活函数
        self.relu = nn.ReLU()
        self.func2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fuc1(x)
        x = self.relu(x)
        y_pred = self.func2(x)
        return y_pred


# 规则：输出最大数值的位置
def Create_rule():
    x = torch.randn(1, 5)
    y = torch.argmax(x, dim=1)
    return x, y

# 准备特征5，分类5的随机数据
def Create_date(input_num):
    # X = []
    # Y = []
    # for i in range(input_num):
    #     x, y = Create_rule()
    #     X.append(x.squeeze())
    #     Y.append(y.item())
    # print(X)
    # print(Y)
    # return torch.FloatTensor(X), torch.LongTensor(Y)

    # 直接生成 (input_num, 5) 的矩阵，一步到位
    X = torch.randn(input_num, 5)
    # 直接计算每一行最大值的索引，得到 (input_num,) 的向量
    Y = torch.argmax(X, dim=1)

    return X.float(), Y.long()

def main():
    # 准备数据参数
    np.random.seed(42)
    num_samples = 1024  # 总数据量
    num_classes = 5  # 5分类
    num_size = 5  # 5个特征

    # 准备训练参数
    num_epochs = 100  # 总训练次数
    batch_size = 32  # 每轮训练的批次
    learn_rate = 0.01  # 学习率

    # 初始化模型
    model = TorchModle(num_size, 32, num_classes)

    # 使用交叉熵计算5分类任务，内置softmax
    Criterion_torch = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    log = []
    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        trainX, trainY = Create_date(num_samples)

        watch_loss = []

        # 打乱数据
        indices = torch.randperm(num_samples)
        x_shuffled = trainX[indices]
        y_shuffled = trainY[indices]

        for batch in range(num_samples // batch_size):
            x = x_shuffled[batch * batch_size : (batch + 1) * batch_size]
            y = y_shuffled[batch * batch_size : (batch + 1) * batch_size]
            y_pred = model(x)
            loss = Criterion_torch(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"=========\n第 {epoch + 1} 轮平均 loss: {avg_loss:.4f}")
        log.append([float(avg_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图
    plt.plot(range(len(log)), log, label="Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_model(model_path):
    # 初始化模型
    model = TorchModle(5, 32, 5)

    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式

    # 准备测试数据
    test_num = 100
    test_x, test_y = Create_date(test_num)

    # 模型预测
    with torch.no_grad():  # 测试阶段不需要计算梯度，节省内存
        output = model(test_x)
        # 获取概率最大的类别索引（即预测的最大值位置）
        pred_y = torch.argmax(output, dim=1)
        # print(f"输入x的值：{test_x}\n 输出预测的值：{pred_y}")

    # 计算准确率
    correct = (pred_y == test_y).sum().item()
    accuracy = correct / test_num
    print(f"测试集准确率: {accuracy:.2%} ({correct}/{test_num})")

    # 绘制对比曲线
    plt.figure(figsize=(10, 6))

    # 绘制真实标签（蓝色圆圈）
    plt.plot(test_y.numpy(), 'o-', label='True Label', color='blue', alpha=0.7)

    # 绘制预测标签（红色叉号）
    plt.plot(pred_y.numpy(), 'x-', label='Predicted Label', color='red', alpha=0.7)

    plt.title(f'Model Test Result (Accuracy: {accuracy:.2%})', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Class Index (0-4)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
    # test_model("model.bin")













