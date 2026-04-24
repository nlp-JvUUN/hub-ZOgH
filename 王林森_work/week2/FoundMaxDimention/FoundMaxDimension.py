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
import torch.optim as optim

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层,输入input_size维向量，输出5列
        self.loss = nn.CrossEntropyLoss() #根据任务需求改成多分类交叉熵


    # 分类前进
    def forward(self, x, y=None):
        # x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1) 错点1
        # y_pred = torch.softmax(x,dim=1)  # dim表示做分类
        # if target is not None:
        #     return self.loss(y_pred, target)  # 预测值和真实值计算损失
        # else:
        #     # print(x)
        #     # print(y_pred)
        #     return y_pred  # 输出预测结果
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 将数据映射到标签上，计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果


# 生成样本与最大值索引集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 随机生成一批样本,构建学习规则
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    print(max_index)
    return x, max_index

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数,保证与目标矩阵形状一致
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.05  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optimer = optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # lrs = optim.lr_scheduler
    # sch = lrs.CosineAnnealingLR(optimer, T_max=5)#余弦退火
    # sch=lrs.StepLR(optimer, step_size=5, gamma=1)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x,y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optimer.step()  # 更新权重
            # sch.step()#更新学习率
            optimer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        log.append([float(np.mean(watch_loss))])# log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="loss")  # 画loss曲线 # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    print("当前输入：", input_vec)
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))#加载训练好的权重
    model.eval()#测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print(res)
        pred_class = res.argmax().item()  # 预测类别
        max_prob = res.max().item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class+1, max_prob))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.222349776,0.99416669,0.92579291,0.41567412,0.1358894],
                [0.666222222,0.77756231,0.14235568,0.89956235,0.43652456]]
    predict("model.bin", test_vec)
