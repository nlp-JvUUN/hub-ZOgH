import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
x是一个5维向量，哪一维数字最大就属于第几类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测概率

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，哪一维数字最大就属于第几类
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 返回最大值的索引
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, num_classes=5):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数量
    class_counts = {i: 0 for i in range(num_classes)}
    for label in y:
        class_counts[int(label)] += 1

    print("本次预测集中各类别样本数量:")
    for i in range(num_classes):
        print(f"  类别{i}: {class_counts[i]}个")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred_prob = model(x)  # 模型预测，返回概率
        y_pred = torch.argmax(y_pred_prob, dim=1)  # 取概率最大的类别

        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 3000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数（5分类任务）
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)  # 传入num_classes

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, num_classes)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_multi.bin")

    # 画图
    print("acc:")
    for i, (acc, loss_val) in enumerate(log):
        print(f"第{i + 1}轮: 准确率={acc:.4f}, 损失={loss_val:.4f}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], 'b-', label="准确率")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], 'r-', label="损失")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec, num_classes=5):
    input_size = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    # 打印模型参数
    print("模型参数:")
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}")

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        probabilities = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        predictions = torch.argmax(probabilities, dim=1)  # 获取预测类别

    print("\n预测结果:")
    for i, (vec, prob, pred) in enumerate(zip(input_vec, probabilities, predictions)):
        true_class = np.argmax(vec)  # 真实类别（最大值索引）
        print(f"样本{i + 1}:")
        print(f"  输入向量: {vec}")
        print(f"  真实类别: {true_class} (第{true_class + 1}维最大, 值={vec[true_class]:.4f})")
        print(f"  预测类别: {int(pred)}")
        print(f"  各类别概率: {[f'{p:.4f}' for p in prob.tolist()]}")
        print(f"  是否正确: {'✓' if int(pred) == true_class else '✗'}")
        print()

if __name__ == "__main__":
    main()

    # 测试数据
    test_vec = [
        [0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.99758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
        [0.1, 0.9, 0.3, 0.4, 0.5],
        [0.2, 0.2, 0.9, 0.2, 0.2],
        [0.3, 0.3, 0.3, 0.9, 0.3],
        [0.4, 0.4, 0.4, 0.4, 0.9],
    ]

    predict("model_multi.bin", test_vec, num_classes=5)
