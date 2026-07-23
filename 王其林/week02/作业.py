"""
第二周作业
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pyplot
import math

# 样本输入向量维度
input_size = 6

# 构建模型
class TorchModel(nn.Module):
    
    def __init__(self, input_size=input_size, hidden_size=8, num_classes=input_size):
        super().__init__()
        # 层1
        self.layer1 = nn.Linear(input_size, hidden_size)  # 线性层
        # 层2
        self.layer2 = nn.Linear(hidden_size, num_classes) # 线性层
        # 不使用激活函数
        # self.activation = nn.ReLU()
        # 损失函数 交叉熵(多分类任务)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y_true=None):
        x = self.layer1(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        logits = self.layer2(x) # (batch_size, hidden_size) -> (batch_size, num_classes)
        if y_true is not None:
            # 内部会计算softmax 再根据交叉熵公式计算损失
            return self.loss(logits, y_true)
        else:
            return logits
        
def build_single_sample():
    """构建单个随机样本"""
    sample = np.random.rand(input_size)
    return sample, np.argmax(sample)

def build_sample_set(sample_size):
    """构建批量样本，用于训练集、验证集、测试集"""
    X = []
    Y = []
    for i in range(sample_size):
        x, y = build_single_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate_model(model: nn.Module):
    """验证模型"""
    # 验证集样本数量
    eval_set_size = 200
    # 构建验证集
    eval_set_x, eval_set_y = build_sample_set(eval_set_size)
    # 预测正确数量，用于计算准确率
    correct = 0
    # 开始验证
    model.eval()
    # 只做推理,不需要计算梯度
    with torch.no_grad():
        # 模型输出
        logits = model(eval_set_x)
        # 将输出转换为预测结果，取最大值索引下标 因为最大值经过softmax函数后也是最大值
        y_pred = torch.argmax(logits, dim=1)
        # 计算预测正确数量
        correct = (y_pred == eval_set_y).sum().item()
    # 计算准确率
    accuracy = correct / eval_set_size
    print(f"验证集数量:{eval_set_size}, 预测正确数量:{correct}, 准确率:{accuracy}")
    return accuracy
        
def main():
    # 超参数
    epoch_size = 60    # 训练轮数
    batch_size = 32     # 每批训练样本数量
    sample_size = 5000  # 每轮训练样本数量
    lr = 0.01           # 学习率

    # 构建训练集
    train_set_x, train_set_y = build_sample_set(sample_size)

    # 构建模型
    model = TorchModel(input_size)

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 日志记录
    log = []

    # 训练过程
    for epoch_idx in range(epoch_size):
        # 开始训练
        model.train()

        # 记录每一轮损失
        epoch_loss = []
        # 每轮训练多少批样本
        batch_count = math.ceil(sample_size / batch_size)
        for batch_idx in range(batch_count):
            if batch_idx < sample_size // batch_size:
                # 从样本中取出每批次需要训练的样本
                x = train_set_x[batch_size * batch_idx : batch_size * (batch_idx + 1)]
                y = train_set_y[batch_size * batch_idx : batch_size * (batch_idx + 1)]
            elif sample_size % batch_size > 0:
                # 剩余的样本
                x = train_set_x[sample_size - sample_size % batch_size :]
                y = train_set_y[sample_size - sample_size % batch_size :]
            # 正向传播 计算损失
            loss = model(x, y)
            # 反向传播 计算梯度
            loss.backward()
            # 根据梯度更新权重
            optim.step()
            # 梯度清零
            optim.zero_grad()

            # 记录每批损失
            epoch_loss.append(loss.item())

        # 验证模型
        accuracy = evaluate_model(model)
        # 记录日志
        log.append([accuracy, np.mean(epoch_loss)])

    # 训练完成 保存模型参数
    file_path = "model.bin" # 文件路径
    torch.save(model.state_dict(), file_path)

    # 根据记录日志画图
    pyplot.plot(range(len(log)), [l[0] for l in log], label="acc")  # acc曲线
    pyplot.plot(range(len(log)), [l[1] for l in log], label="loss") # loss曲线
    pyplot.legend()
    pyplot.show()    

def predict(file_path):
    """模型预测"""
    # 测试集样本数量
    test_set_size = 10
    # 构建测试集
    test_set_x, test_set_y = build_sample_set(test_set_size)
    # 构建模型
    model = TorchModel()
    # 从本地加载模型参数
    model.load_state_dict(torch.load(file_path))
    # 模型预测
    model.eval()
    # 只做推理,不需要计算梯度
    with torch.no_grad():
        # 模型输出
        digits = model(test_set_x)
        # 将输出转换为预测结果，取最大值索引下标 因为最大值经过softmax函数后也是最大值
        y_pred = torch.argmax(digits, dim=1)
        # 将输出经过softmax转换为概率值
        y_p = torch.softmax(digits, dim=1)
        
        for x, y, y_p, y_true in zip(test_set_x, y_pred, y_p, test_set_y):
            print(f"数据:{x} 预测值:{y} 预测概率:{y_p.tolist()} 是否正确:{y_true == y}")

if __name__ == "__main__":
    main()
    predict("model.bin")
