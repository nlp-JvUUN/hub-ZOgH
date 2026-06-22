#完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维数字最大就属于第几类，例如[1,2,3,4,5]，第3维数字最大，就属于第3类。

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)
        self.activation = nn.Softmax(dim=1) # 激活函数
        self.loss = nn.CrossEntropyLoss() # 损失函数
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

# 生成样本，随机生成一个5维向量，哪一维数字最大就属于第几类
def build_sample():
    x = np.random.random(5)
    max_index = 0 
    for i in range(5):
        if x[i] > x[max_index]:
            max_index = i
    y = max_index
    return x, y         

#print(build_sample())

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    #print(X)
    #print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

#print(build_dataset(5))

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    class_count = [0, 0, 0, 0, 0]
    for lable in y:
        class_count[int(lable)] += 1
    print("本次预测集中共有%d个第1类样本，%d个第2类样本，%d个第3类样本，%d个第4类样本，%d个第5类样本" % (class_count[0], class_count[1], class_count[2], class_count[3], class_count[4]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测 model.forward(x)
        for i in range(len(y_pred)): # 与真实标签进行对比
            y_p = y_pred[i]
            y_t = y[i]
            max_index = 0
            for j in range(5):
                if y_p[j] > y_p[max_index]:
                    max_index = j
            #判断预测是否正确
            if max_index == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 50 # 训练轮数
    batch_size = 50 # 每次训练样本个数
    train_sample = 10000 # 每轮训练总共训练的样本总数
    input_size = 5 # 输入向量维度
    learning_rate = 0.005 # 学习率
    class_num = 5 # 类别数量
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 模型训练
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss  model.forward(x,y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))    
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_xdq1.bin")
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
    class_num =5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        max_index = 0
        max_prob = res[0]
        for i in range(1,5):
            if res[i] > max_prob:
                max_prob = res[i]
                max_index = i

        predicted_class = max_index
        print("输入：%s, 预测类别：%d, 最大概率值：%f" % (vec, predicted_class, max_prob))

if __name__ == "__main__":
    main()
    test_vec = [[0.8, 0.78, 0.82, 0.79, 0.81],
                [0.6, 0.61, 0.59, 0.62, 0.58],
                [0.95, 0.89, 0.91, 0.92, 0.9],
                [0.4, 0.41, 0.39, 0.42, 0.4],
                [0.73, 0.71, 0.69, 0.72, 0.7]]
    predict("model_xdq1.bin", test_vec)


