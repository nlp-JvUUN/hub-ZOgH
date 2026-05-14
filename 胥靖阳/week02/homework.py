# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : homework
@Author : lycoris
@Time : 2026/4/22 18:49  
@脚本说明 : 

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# DataLoader 数据加载器，将数据转换为TensorData格式
# TensorDataset 数据设置器，将数据转换为TensorData格式
from torch.utils.data import DataLoader, TensorDataset

DIM = 5  # 向量维度（也是类别数）
NUM_SAMPLES = 5000      # 样本总数
BATCH_SIZE = 50     # 单次样本数
EPOCHS = 100     # 训练轮数
LR = 0.001       # 学习率

# 损失函数
criterion = nn.CrossEntropyLoss()    # 使用 CrossEntropyLoss

class IndexTrainModel(nn.Module):
    def __init__(self, input_size, hidden_size=[64,32]):
        super().__init__()
        layers = []
        buffercount = input_size# 神经网络层
        for size in hidden_size:                       # 循环hidden_size创建神经网络，size：维度
            layers.append(nn.Linear(buffercount, size))
            layers.append(nn.ReLU())                # ReLU它快
            buffercount = size
        layers.append(nn.Linear(buffercount, input_size))       # 输出是各个维度是最大值的可能
        # nn.Sequential  顺序排列神经网络
        # *layers        *python的参数解包字符，等价于 nn.Sequential(Linear(...), ReLU(), Linear(...))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)      # 返回 logits，不经过 Softmax

def build_dataset():
    #生成数据集
    X = torch.randn(NUM_SAMPLES, DIM)  # 输入：随机向量
    y = torch.argmax(X, dim=1) # 标签：最大值所在的索引

    # 划分训练集和测试集
    split = int(0.8 * NUM_SAMPLES)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    return train_loader, test_loader

def evaluate(model, test_loader):
    model.eval()
    correct, wrong = 0, 0
    # no_grad禁用梯度计算，不构建计算图
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            y_t = torch.argmax(logits, dim=1)
            length = y_t.size(0)
            for i in range(length):
                if y_t[i].item() == y[i].item():
                    correct += 1
                else:
                    wrong += 1
        print("正确预测个数：%d, 总数：%d，正确率：%f" % (correct, (correct + wrong), correct / (correct + wrong)))
        return correct / (correct + wrong)

def main():
    train_loader, test_loader = build_dataset()
    model = IndexTrainModel(DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad() # 梯度归零
            logitems = model(X)
            loss = criterion(logitems, y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step() # 更新权重
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {acc:.4f}")

if __name__ == '__main__':
    main()