import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务：
输入一个n维向量，输出最大值的索引（0 ~ n-1）作为类别标签
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出维度 5
        self.loss = nn.CrossEntropyLoss()    # 交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值（logits或概率）
    def forward(self, x, y=None):
        y_pred = self.linear(x)# (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)      # 计算交叉熵损失，y为LongTensor类型
        else:
            # 推理阶段返回概率分布（softmax后的结果）和预测类别  softmax作用是把向量里的值变为0到1的取值
            probs = nn.functional.softmax(y_pred, dim=-1)
            return probs

# 生成一个样本：随机生成一个n维向量，最大值所在索引即为类别标签
def build_sample():
    x = np.random.random(5)
    maxValue = x[0]
    maxIndex = 0
    for index, value in enumerate(x):
        if value > maxValue:
            maxValue = value
            maxIndex = index
    return x, maxIndex

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)   # 因为Y是整数

# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x) #获取到模型的预测值
        for y_p, y_t in zip(y_pred, y):
            arr = y_p.numpy()
            max_index = np.argmax(y_p)
            print("最大值索引:", max_index)  # 输出 4
            print("最大值:", arr[max_index])
            if max_index == y_t:
                correct+=1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20           # 训练轮数
    batch_size = 20          # 每次训练样本个数
    train_sample = 5000      # 每轮训练总共训练的样本总数
    input_size = 5           # 输入向量维度（同时决定类别数）
    learning_rate = 0.01     # 学习率

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    print(f"训练集样本数：{len(train_x)}，输入维度：{input_size}，类别数：{input_size}")

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)               # 计算loss
            loss.backward()                  # 计算梯度
            optim.step()                     # 更新权重
            optim.zero_grad()                # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)    # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model_jcs.bin")
    print("模型已保存为 model_jcs.bin")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    # 输入向量的维度决定类别数
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        probs = model(input_tensor)  # 概率分布  因为没有Y 输出的都是softmax后的值
        preds = torch.argmax(probs, dim=1)  # 预测类别 --> 取出向量里最大的索引向量
        print(f"preds={preds}")
    print("\n========= 预测结果 ===begin======")
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        # 打印输入向量、预测类别、每个类别的概率
        prob_list = prob.tolist()
        print(f"样本{i + 1}: 输入{input_vec[i]}")
        #将数字 p 四舍五入到小数点后 3 位。例如
        print(f"       预测类别: {pred}, 各类概率: {[round(p, 3) for p in prob_list]}")
    print("\n========= 预测结果 ===end======")

if __name__ == "__main__":
    # main()
    # 测试预测功能（可选）
    test_vec = [
        [0.1, 0.9, 0.2, 0.3, 0.4],   # 最大值索引1
        [0.8, 0.2, 0.7, 0.6, 0.5],   # 最大值索引0
        [0.3, 0.4, 0.8, 0.2, 0.1],   # 最大值索引2
        [0.5, 0.5, 0.5, 0.9, 0.4]    # 最大值索引3
    ]
    predict("model_jcs.bin", test_vec)