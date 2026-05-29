# coding:utf8

# 解决 OpenMP 库冲突问题
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，五维随机向量最大的数字在哪维就属于哪一类，
比如x=[0.1, 0.2, 0.3, 0.4, 0.5]，则x属于第5类

实现思路：
只有一个线性层，加交叉熵自带的softmax作为激活
"""


class TorchMultiClsModel(nn.Module):  
    def __init__(self, input_size, num_classes):
        super(TorchMultiClsModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        # 交叉熵自带softmax，所以这里不需要再加一个softmax层了
        # self.activation = nn.Softmax(dim=1)  # nn.Softmax() softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失
        # self.loss = nn.functional.cross_entropy  # 也可以直接使用函数式接口  这两种写法都可以

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)
        return logits


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，五维随机向量最大的数字在哪维就属于哪一类
# 比如 函数返回：
# x = [0.3， 0.2， 0.5， 0.1， 0.4]，则返回 x 和 类别3（从0开始数）
def build_sample(dim=5):
    x = np.random.random(dim)
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
# 使用from_numpy将numpy数组转换为torch张量, 比torch.floattensor更快
def build_dataset(total_sample_num, dim=5):
    x = np.random.random((total_sample_num, dim)).astype(np.float32)
    y = np.argmax(x, axis=1).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        logits = model(x)           # 模型预测 model.forward(x)
        y_pred = torch.argmax(logits, dim=1)
    correct = (y_pred == y).sum().item()
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num     = 100     # 训练轮数
    batch_size    = 20      # 每次训练样本个数
    train_sample  = 5000    # 每轮训练总共训练的样本总数
    input_size    = 5       # 输入向量维度
    learning_rate = 0.0001  # 学习率
    # 建立模型
    model = TorchMultiClsModel(input_size, num_classes=5)  # 假设有5个类别
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 用来存储每轮训练的准确率和loss值，画图用
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 这个应该写在循环内还是外？
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
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
    model = TorchMultiClsModel(input_size, num_classes=5)  # 假设有5个类别  
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        input_tensor = torch.as_tensor(np.asarray(input_vec), dtype=torch.float32)
        logits = model(input_tensor)  # 模型预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    for vec, pred, prob in zip(input_vec, preds, probs):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(pred), float(prob[int(pred)])))  # 打印结果


if __name__ == "__main__":
    main()
    datademo = build_dataset(100)
    print(datademo)
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)
