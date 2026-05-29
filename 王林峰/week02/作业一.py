# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
改造为多分类任务
规律：x是一个5维向量，哪一维数字最大，就属于第几类（0~4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出维度=类别数
        self.loss = nn.CrossEntropyLoss()  # 多分类用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测概率
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出各类别概率

# 生成一个样本：5维向量，哪一维最大，标签就是几
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 取最大值所在的索引作为标签
    return x, y

# 生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签用LongTensor

# 测试代码：测试每轮准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
        correct = (y_pred == y).sum().item()
        wrong = test_sample_num - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 20       # 每次训练样本数
    train_sample = 5000   # 总样本数
    input_size = 5        # 输入维度
    num_classes = 5       # 类别数量（5分类）
    learning_rate = 0.01  # 学习率
    
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_multi_class.bin")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        true_class = np.argmax(vec)
        print(f"输入：{vec}\n真实类别：{true_class}, 预测类别：{pred_class}, 各类概率：{res.numpy()}\n")

if __name__ == "__main__":
    main()
    # 测试用例
    test_vec = [
        [0.1, 0.2, 0.8, 0.3, 0.2],   # 最大是第3位（索引2）
        [0.9, 0.1, 0.2, 0.1, 0.1],   # 最大是第1位（索引0）
        [0.2, 0.7, 0.1, 0.2, 0.1],   # 最大是第2位（索引1）
        [0.1, 0.1, 0.1, 0.9, 0.1],   # 最大是第4位（索引3）
        [0.2, 0.1, 0.1, 0.1, 0.8]    # 最大是第5位（索引4）
    ]
    predict("model_multi_class.bin", test_vec)