import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于 pytorch 框架编写模型训练
规律：x 是一个 5 维向量，哪一维数字最大就属于第几类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size) # 线性层
        # self.activation = torch.softmax # nn.Softmax() softmax 激活函数
        self.loss = nn.functional.cross_entropy # loss 函数采用交叉熵损失

    def forward(self, x, y = None):
        x = self.linear(x)
        # y_pred = self.activation(x, dim = 0)
        if y is not None:
            # return self.loss(y_pred, y) # 计算损失
            return self.loss(x, y)
        else:
            return x

def build_sample():
    """
    生成一个样本
    随机生成一个 5 维向量，哪一维数字最大就属于第几类
    """
    x = np.random.random(5)
    label = np.argmax(x)
    return x, label

def build_dataset(total):
    """随机生成一批样本"""
    X = []
    Y = []
    for i in range(total):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)

def evaluate(model):
    """测试模型的准确率"""
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测 model.forward(x)
        y_pred_class = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_class, y): # 与真实值进行对比
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练使用的样本数
    input_size = 5 # 输入向量的维度
    learning_rate = 0.01 # 学习率

    # 建立模型
    model = TorchModel(input_size, input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 切片，每次取 20 个数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算 loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
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

def predict(model_path, input_vec):
    """使用训练好的模型做预测"""
    input_size = 5
    model = TorchModel(input_size, input_size)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重
    print(model.state_dict())

    model.eval()
    with torch.no_grad(): # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec)) # 执行模型预测
        result_class = torch.argmax(result, dim = 1)
    
    for vec, res in zip(input_vec, result_class):
        print("输入：%s, 预测类别：%d" % (vec, int(res)))  # 打印结果

if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)