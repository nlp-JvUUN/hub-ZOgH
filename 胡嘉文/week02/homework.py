"""
第二周作业：
尝试完成一个多分类任务的训练：一个随机向量，哪一维数字最大就属于第几类
"""
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return torch.softmax(x, dim=1)


# 样本生成
def build_sample(num_samples, dim):
    # 1. 一次性生成 num_samples 行，dim 列的随机数矩阵
    # 形状：(num_samples, dim)
    X = np.random.random((num_samples, dim))
    
    # 2. 沿着列的方向（axis=1）找最大值的索引
    # 形状：(num_samples,)，里面装的全是 0 到 dim-1 的整数
    Y = np.argmax(X, axis=1)
    
    return X, Y

def build_dataset(num_samples, dim=5):
    X, Y = build_sample(num_samples, dim)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model, input_size=5):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        pred_classes = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(pred_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


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
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
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

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        true_class = np.argmax(vec)
        print("输入：%s, 真实类别：%d, 预测类别：%d, 概率值：%f" % (vec, true_class, pred_class, res[pred_class]))


if __name__ == "__main__":
    # 测试代码
    # main()
    # 测试预测
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)