import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
'''
多分类任务
一个随机向量，哪一维的数字最大，就属于第几类
例如：一个3维向量[2, 5, 3], 向量值5最大下标为1，则属于第1类
损失函数：交叉熵
'''
input_size = 5 # 一个随机向量的维度
num_classes = 5 # 分的几类任务
class TorchModel(nn.Module):
    def __init__(self, input_size = 5, num_classes = 5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss() # 交叉熵函数
    def forward(self, x, y = None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.long()) # y必须是long type的，不能是float 类型的
        else:
            return y_pred

# 生成随机向量以及获取随机向量中最大向量的下标
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

# print(build_sample())

def build_dataes(total_sample_num = 100):
    '''
    批量生成一批样本
    :param total_sample_num:
    :return:
    '''
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 模型测试
def evaluate(model):
    model.eval()
    print(f"\n{'=' * 30} 模型测试 {'=' * 30}")
    with torch.no_grad():
        test_vec = np.random.random(5)
        test_tensor = torch.FloatTensor(test_vec)
        outPut = model(test_tensor)
        # print(outPut)
        pred_cls = torch.argmax(outPut).item()
        # print(pred_cls)
        true_cls = np.argmax(test_vec)
        print(f"测试向量：{test_vec}")
        print(f"真实的分类：{true_cls}")
        print(f"训练预测的分类：{pred_cls}")

# build_dataes(2)
def main():
    '''
    主程序
    :return:
    '''
    epoch_num = 20  # 训练轮数
    batch_size = 100  # 每次训练的样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 选择训练集
    train_x, train_y = build_dataes(train_sample)
    for epoch in range(epoch_num):
        model.train() # 开始训练
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss
            loss.backward() # 计算梯度 反向传播
            optim.step()  # 更新参数
            optim.zero_grad() # 清空参数
            watch_loss.append(loss.item())
        print(f"===========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")

    # 测试模型
    evaluate(model)
    return

if __name__ == "__main__":
    main()

