import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5维，对应5个类别
        self.loss = nn.CrossEntropyLoss()  # CrossEntropyLoss 内部已经包含了 softmax

    def forward(self, x, y=None):
        x = self.linear(x)  # 输出 (batch_size, 5)
        if y is not None:
            return self.loss(x, y)  # CrossEntropyLoss 直接接收 logits
        else:
            return F.softmax(x, dim=1)  # 预测时返回概率分布


# 生成一个样本
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 输出概率分布 (100, 5)
        y_pred = torch.argmax(y_pred, dim=1)  # 取概率最大的索引作为预测类别
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 100
    batch_size = 40
    train_sample = 10000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.bin")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        # res 是5维概率向量，取最大值的索引
        pred_class = int(torch.argmax(res))
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, pred_class, res))


if __name__ == "__main__":
    # main()
    test_vec = [
        [0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.20797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
    ]
    predict("model.bin", test_vec)
