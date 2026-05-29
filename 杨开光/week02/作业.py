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
规律：x是一个5维向量，每个数都是一个类别，找出最大数对应的索引作为类别

"""


class TorchModel(nn.Module):
    """
    TorchModel：自定义的PyTorch模型

    模型结构：
        线性层 (Linear): 输入5维向量，输出5个类别的logits
        激活函数 (Softmax): 将logits转换为概率分布（多分类常用）

    前向传播流程：
        1. 线性层: y = Wx + b
        2. Softmax: 将5个logits转换为5个概率值（所有概率和为1）
    """

    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 线性层：输入维度为input_size(5)，输出维度为5（对应5个类别）
        # 权重W shape: (5, input_size)，偏置b shape: (5,)
        self.linear = nn.Linear(input_size, 5)

        # Softmax激活函数：在第1维度（类别维度）上进行归一化
        # 将5个logits转换为5个概率值，每个值在0~1之间，且总和为1
        self.softmax = nn.Softmax(dim=1)

        # 损失函数：交叉熵损失（CrossEntropyLoss）
        # 用于衡量预测概率分布与真实标签之间的差异
        self.loss = nn.functional.cross_entropy_loss

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 线性变换：输入x通过线性层
        # 输入shape: (batch_size, input_size) = (batch_size, 5)
        # 输出shape: (batch_size, 5)，每个样本得到5个类别的logits（未归一化分数）
        x = self.linear(x)

        # Softmax激活：将logits转换为概率分布
        # 例如 logits = [2.0, 1.0, 0.1, -1.0, 0.5]
        #         → softmax → [0.74, 0.27, 0.11, 0.04, 0.17]（概率和为1）
        x = self.softmax(x)

        # 如果提供了真实标签y，计算损失用于训练
        if y is not None:
            # 注意：CrossEntropyLoss内部包含Softmax，所以通常这里不需要加
            # 但为了让predict时输出明确的概率值，这里保留了Softmax
            # y.squeeze()将y从shape (batch_size, 1) 转为 (batch_size,)
            return self.loss(x, y.squeeze())
        else:
            # 无真实标签时，返回预测的概率分布
            # 输出shape: (batch_size, 5)，每行是5个类别的概率值
            return x


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值对应的索引作为类别标签
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值的索引作为类别
    return x, y


# 随机生成一批样本
# 5个类别均匀生成
def build_dataset(total_sample_num):
    """
    构建数据集：生成指定数量的样本

    工作流程：
    1. 循环调用 build_sample() 函数 total_sample_num 次
    2. 每次生成一个5维输入向量x和对应的类别标签y
    3. 将所有样本的x和y分别收集到列表X和Y中
    4. 将列表转换为PyTorch张量并返回

    参数:
        total_sample_num: 需要生成的样本数量
    返回:
        x: FloatTensor类型的输入向量，shape为 (total_sample_num, 5)
        y: LongTensor类型的类别标签，shape为 (total_sample_num,)
    """
    X = []  # 用于存储所有输入向量的列表
    Y = []  # 用于存储所有类别标签的列表

    # 循环生成total_sample_num个样本
    for i in range(total_sample_num):
        # 调用build_sample()生成一个样本
        # x: 5维numpy数组，输入向量
        # y: 整数(0~4)，该向量最大值的索引作为类别标签
        x, y = build_sample()

        # 将输入向量x添加到列表X中
        X.append(x)

        # 将类别标签y添加到列表Y中
        Y.append(y)

    # 打印生成的输入和标签（可选，用于调试）
    # print(X)
    # print(Y)

    # 将Python列表转换为PyTorch张量
    # torch.FloatTensor(X): 将X列表转换为FloatTensor类型
    #   shape: (total_sample_num, 5)，每一行是一个5维输入向量
    # torch.LongTensor(Y): 将Y列表转换为LongTensor类型
    #   shape: (total_sample_num,)，每个元素是一个类别标签(0~4的整数)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    """
    评估函数：测试每轮训练后模型的准确率

    工作流程：
    1. 将模型切换到评估模式（model.eval()），关闭dropout等训练特定层
    2. 生成100个测试样本（x:输入向量, y:真实类别标签）
    3. 使用模型对测试集进行预测
    4. 比较预测类别和真实类别，计算正确率

    参数:
        model: 训练好的PyTorch模型
    返回:
        float: 预测准确率（0~1之间）
    """
    model.eval()  # 评估模式，关闭 dropout、batchnorm 等训练层

    test_sample_num = 100  # 生成100个测试样本

    # 调用build_dataset生成测试集
    # 真实类别标签y的来源：
    #   1. build_sample()函数随机生成一个5维向量 x = np.random.random(5)
    #   2. 找出x中最大值的索引: y = np.argmax(x)，即最大值的位置(0~4)
    #   3. 例如 x = [0.2, 0.8, 0.1, 0.5, 0.3]，最大值是0.8，索引是1，则y=1
    # x: shape为(100,5)的输入向量，y: shape为(100,)的真实类别标签(0~4)
    x, y = build_dataset(test_sample_num)

    print("本次预测集中共有5个类别")

    # 初始化正确和错误预测的计数器
    correct, wrong = 0, 0

    # with torch.no_grad(): 临时关闭梯度计算，节省内存和计算资源
    with torch.no_grad():
        # 模型预测：输入测试集x，输出每个样本属于5个类别的分数(logits)
        # 预测类别 y_pred_class 的来源：
        #   1. model(x) 将测试输入x传入训练好的模型
        #   2. 模型内部计算：y_pred = linear(x)，输出shape为(100, 5)的 logits
        #   3. logits表示每个样本属于5个类别的"原始分数"（未经归一化）
        #   4. 使用argmax在类别维度(dim=1)上取最大分数的索引，即为预测类别
        # y_pred shape: (100, 5)，每行表示一个样本对应5个类别的未归一化分数
        y_pred = model(x)

        # torch.argmax(y_pred, dim=1): 在第1维度(类别维度)上找到最大值的索引
        # 得到每个样本预测的类别（0~4）
        # 例如 y_pred = [[2.1, 0.3, -1.2, 0.5, 0.1], ...]，第0个分数最大，则预测类别为0
        # y_pred_class shape: (100,)
        y_pred_class = torch.argmax(y_pred, dim=1)

        # 遍历每个样本的预测类别和真实类别
        # zip(y_pred_class, y) 将预测结果和真实标签配对
        for y_p, y_t in zip(y_pred_class, y):
            # 比较预测类别(y_p)和真实类别(y_t)是否相同
            if int(y_p) == int(y_t):
                correct += 1  # 预测正确，计数器+1
            else:
                wrong += 1   # 预测错误，计数器+1

    # 打印本轮的预测正确个数和准确率
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))

    # 返回准确率，用于记录训练过程中的模型性能变化
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型：创建TorchModel实例
    # 模型包含一个线性层（输入5维→输出5维）和Softmax激活
    model = TorchModel(input_size)

    # 选择优化器：Adam优化器
    # Adam是一种自适应学习率的优化算法，结合了动量和RMSprop的优点
    # 参数：
    #   model.parameters(): 模型的所有可学习参数（权重W和偏置b）
    #   lr=learning_rate: 学习率，控制权重更新的步长（这里为0.01）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 用于记录每轮的准确率和损失
    log = []

    # 创建训练集：生成5000个样本
    # train_x: shape (5000, 5) 的输入向量
    # train_y: shape (5000,) 的类别标签
    train_x, train_y = build_dataset(train_sample)

    # 训练过程：共训练20轮（epoch）
    for epoch in range(epoch_num):
        # 切换到训练模式（启用dropout等训练层）
        model.train()

        # 记录每轮的loss
        watch_loss = []

        # 每个epoch内，将样本分成多个batch进行训练
        # 5000 / 20 = 250个batch
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch的数据（20个样本）
            # 例如 batch_index=0: x[0:20], y[0:20]
            #      batch_index=1: x[20:40], y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            # 第1步：前向传播
            # 将batch数据传入模型，计算loss
            # model.forward(x, y) 返回交叉熵损失
            loss = model(x, y)

            # 第2步：反向传播
            # loss.backward() 计算损失相对于每个参数的梯度
            # 梯度存储在每个参数的 .grad 属性中
            loss.backward()

            # 第3步：更新权重
            # optim.step() 根据计算出的梯度更新模型参数
            # Adam优化器会自适应地调整每个参数的学习率
            optim.step()

            # 第4步：梯度归零
            # optim.zero_grad() 清空梯度，为下一个batch做准备
            # 如果不归零，梯度会累积
            optim.zero_grad()

            # 记录本batch的loss值
            watch_loss.append(loss.item())

        # 打印本轮的平均loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 评估本轮模型的准确率
        acc = evaluate(model)

        # 记录准确率和loss到日志
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
    """
    预测函数：使用训练好的模型对新数据进行预测

    工作流程：
    1. 创建模型结构（TorchModel）
    2. 从文件加载训练好的权重参数（model.load_state_dict）
    3. 切换到评估模式（model.eval）
    4. 将输入转换为PyTorch张量并进行预测
    5. 使用argmax获取预测的类别

    参数:
        model_path: 保存模型权重的文件路径（.bin文件）
        input_vec: 输入向量列表，每个向量是5维的numpy数组
    返回:
        无直接返回值，结果通过print输出
    """
    input_size = 5  # 输入向量维度为5

    # 创建模型结构（输入5维，输出5个类别）
    model = TorchModel(input_size)

    # torch.load(model_path): 从文件加载保存的模型权重
    # model.load_state_dict(...): 将加载的权重参数加载到模型中
    # 这个权重是在训练过程中通过torch.save保存的
    model.load_state_dict(torch.load(model_path))

    # 打印模型当前的权重参数（可选，用于调试）
    print(model.state_dict())

    # 切换到评估模式，关闭dropout等训练层
    model.eval()

    # with torch.no_grad(): 预测时不计算梯度，节省内存
    with torch.no_grad():
        # 将输入的Python列表转换为PyTorch的FloatTensor张量
        # shape: (n, 5)，n为输入向量的个数
        # 模型前向传播：输出每个样本属于5个类别的分数(logits)
        # result shape: (n, 5)
        result = model.forward(torch.FloatTensor(input_vec))

        # torch.argmax(result, dim=1): 在类别维度上取最大分数的索引
        # 即为预测的类别（0~4）
        # pred_class shape: (n,)
        pred_class = torch.argmax(result, dim=1)

    # 遍历每个输入向量及其预测结果并打印
    for vec, res in zip(input_vec, pred_class):
        # vec: 输入的5维向量
        # res: 模型预测的类别（0~4）
        print("输入：%s, 预测类别：%d" % (vec, int(res)))


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
