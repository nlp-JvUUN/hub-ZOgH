
# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：创建一个随机向量，完成多分类任务，哪一维数字最大就属于第几类
多分类任务：每个样本属于且仅属于一个类别 1D 比如:shape(3,)
多标签分类：每个样本可以属于多个类别2D 比如shape(3,2)
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()#调用父类的初始化方法
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数 激活函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.loss=nn.CrossEntropyLoss()


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失#交叉熵内部存在sigmoid函数
        else:
            return x  # 输出预测结果

def build_sample():
    x = np.random.random(5) #生成的是numpy数组 x=array([])
    category = np.argmax(x) # 找到最大值的索引作为类别（0-4共5类）
    return x, category

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        print(x,y)
        X.append(x) 
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)#

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    y_true = y # (100, 1) -> (100,)
    print("本次预测集中共有5个类别，样本均匀分布")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # model.forward(x)
        y_pred_idx = torch.argmax(y_pred, dim=1)  # 找最大值的索引
        for pred, true in zip(y_pred_idx, y_true):
            if int(pred) == int(true):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000 # 总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size].squeeze()  # (20, 1) -> (20,)
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        result_idx = torch.argmax(result, dim=1)
        result_prob = torch.softmax(result, dim=1)
    for vec, idx, prob in zip(input_vec, result_idx, result_prob):
        print("输入：%s, 预测类别：%d, 各类别概率：%s" % (vec, int(idx), ['%.2f' % p for p in prob.tolist()]))


if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)



"""
结果:输入：[0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843], 预测类别：0, 各类别概率：['0.53', '0.00', '0.01', '0.00', '0.45']
输入：[0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681], 预测类别：2, 各类别概率：['0.24', '0.02', '0.32', '0.27', '0.14']
输入：[0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392], 预测类别：0, 各类别概率：['0.80', '0.16', '0.01', '0.02', '0.01']
输入：[0.99349776, 0.59416669, 0.99579291, 0.41567412, 0.1358894], 预测类别：2, 各类别概率：['0.45', '0.04', '0.50', '0.01', '0.00']
"""
