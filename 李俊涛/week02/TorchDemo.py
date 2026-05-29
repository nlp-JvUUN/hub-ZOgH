import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) # 线性层
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            # 使用softmax,需要两个参数，输入数据，对哪一维做归一化处理
            return torch.softmax(y_pred,dim=1)


def build_sample():
    x = np.random.random(5)  # 生成一个一维的5个数
    return x, np.argmax(x)


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # 如果不加[] 那么利用FloatTensor将数据转换后，成一维张量，[batch_size,]
        # 但是我们想要的是[batch_size,1] 即转化为二维数据，
        Y.append(y)

    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    # 通常搭配 with torch.no_grad 使用，不采用梯度计算，但如果调用
    # loss.backward() or optim.step()函数，梯度值依旧会更新
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print('本次预测集中共有%d个正样本，%d个负样本' % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print('正确预测个数： %d，正确率 %f' % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 50
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        # 开启训练模式，与eval模式的区别在于在特定层的逻辑不同
        # 例如 Dropout层 and Batch Normalization层的计算逻辑不同
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 模型定义中 定义返回loss model(x,y) 不仅返回loss，还会返回
            # 从y_pred 到 x的计算图 供反向传播使用
            loss = model(x, y)
            # 沿着计算图计算各个需要梯度的值， 即requires_grad = True 的值
            # 在本模型中self.linear.weight and self.linear.bias 被自动标记
            # 计算出梯度后并不会自动更新 梯度 即如果不清洗梯度，梯度会进行累加
            # 好处：在显存不足时，可以利用该机制达到大batch_size的效果
            loss.backward()
            # 根据每个参数的.grad存储的梯度，按照自身优化器的规则进行梯度计算
            optim.step()
            # 梯度清零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print('\n第%d轮平均loss：%f' % (epoch +1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 模型保存，保存一个字典对象，里面有weight，bias等学习参数
    # 使用时，必须使用相同的模型结构，
    torch.save(model.state_dict(), 'model.bin')

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res)
        print(f'输入： {vec}，预测类别：{pred_class}概率值：{res}' )

if __name__ == '__main__':
    # main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict('model.bin', test_vec)
