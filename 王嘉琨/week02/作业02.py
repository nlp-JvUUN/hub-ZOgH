import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


# 定义多分类模型
class MultiClassTorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassTorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return torch.softmax(x, dim = 1)


# 生成一个样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[1]:
        return x, 0
    elif x[0] < x[1] and x[0] > x[2]:
        return x, 1
    else:
        return x, 2


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码，评估每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_counts = np.bincount(y.numpy())
    print(f"本次预测集中类别0有{class_counts[0] if len(class_counts) > 0 else 0}个样本，"
          f"类别1有{class_counts[1] if len(class_counts) > 1 else 0}个样本，"
          f"类别2有{class_counts[2] if len(class_counts) > 2 else 0}个样本")
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        correct = (predicted == y).sum().item()
    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    hidden_size = 10
    num_classes = 3
    learning_rate = 0.01
    # 建立模型
    model = MultiClassTorchModel(input_size, hidden_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 10
    num_classes = 3
    model = MultiClassTorchModel(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        _, predicted = torch.max(result.data, 1)
    for vec, pred in zip(input_vec, predicted):
        print(f"输入：{vec}, 预测类别：{pred.item()}")
    return


if __name__ == "__main__":
    main()
    test_vec = [np.random.random(5) for _ in range(5)]
    predict("multi_class_model.bin", test_vec)
