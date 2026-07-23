# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维数字最大就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.activation = torch.softmax  # softmax归一化函数
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            y_pred = self.activation(x, dim=1)  # 应用softmax
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，哪一维数字最大就属于第几类
def build_sample(input_size):
    x = np.random.random(input_size)
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
# 各类别均匀生成
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    print("本次预测集中各类别分布:")
    for i in range(torch.max(y).item() + 1):
        count = (y == i).sum().item()
        print(f"类别{i}: {count}个")
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.01  # 学习率
    
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, input_size)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, input_size)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")
    
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.title("多分类任务训练曲线")
    plt.xlabel("轮数")
    plt.ylabel("值")
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, input_size, num_classes):
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("模型权重:")
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        max_value = vec[predicted_class]
        print(f"输入：{vec}, 预测类别：{predicted_class}, 概率值：{res[predicted_class]:.4f}, 最大值：{max_value:.4f}")  # 打印结果


if __name__ == "__main__":
    main()
    # 测试预测
    test_vec = [
        [0.1, 0.2, 0.8, 0.4, 0.5],  # 第2维最大
        [0.9, 0.2, 0.3, 0.4, 0.5],  # 第0维最大
        [0.1, 0.9, 0.3, 0.4, 0.5],  # 第1维最大
        [0.1, 0.2, 0.3, 0.9, 0.5],  # 第3维最大
        [0.1, 0.2, 0.3, 0.4, 0.9]   # 第4维最大
    ]
    predict("multi_class_model.bin", test_vec, input_size=5, num_classes=5)
