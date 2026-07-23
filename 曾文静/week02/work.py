#第二周作业
#尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类
#多分类任务采用交叉熵损失
#
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  
        self.loss = nn.functional.cross_entropy  #多分类用交叉熵损失函数
    
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果

#随机生成一个5维向量 返回样本和最大值的标签
def build_sample(dim):
    x = np.random.random(dim)
    max_index = np.argmax(x)
    return x, max_index

#构建一批样本
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


  
def evaluate(model, input_size):
    model.eval() #设置为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    correct, wrong = 0, 0
    with torch.no_grad(): #不计算梯度
        y_pred = model(x) #模型预测
        pred_label = torch.argmax(y_pred, dim=1) #获取预测结果的最大值索引
        correct = (pred_label == y).sum().item() #统计正确预测的数量
        wrong = test_sample_num - correct #统计错误预测的数量
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, input_size)
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(len(train_x) // batch_size):    
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y) #计算损失
            loss.backward() #计算梯度
            optim.step() #更新权重
            optim.zero_grad() #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, input_size)
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
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果

if __name__ == '__main__':# 设置超参数
    # main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
