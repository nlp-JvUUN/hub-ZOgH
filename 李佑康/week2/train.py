import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample(input_size):
    x = np.random.random(input_size)
    max_index = np.argmax(x)
    return x, max_index


def build_dataset(total, input_size):
    X = []
    Y = []
    for i in range(total):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model: TorchModel, input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, input_size)
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print("本次预测集各类别样本数：", dict(zip(unique, counts)))
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, pred = torch.max(y_pred, 1)
        correct = (pred == y).sum().item()
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01
    
    model = TorchModel(input_size)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample, input_size)
    
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
        
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, input_size)
        log.append([acc, float(avg_loss)])
        
    torch.save(model.state_dict(), "model.bin")
    
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()  
    
    dim = 5
    test_vec = [[0.1, 0.9, 0.3, 0.2, 0.4],
                  [0.5, 0.3, 0.8, 0.2, 0.1],
                  [0.2, 0.1, 0.3, 0.7, 0.4],
                  [0.9, 0.1, 0.2, 0.3, 0.5],
                  [0.3, 0.4, 0.2, 0.1, 0.8]]
    
    model = TorchModel(dim)
    model.load_state_dict(torch.load("model.bin")) 
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(test_vec))
        _, predicted = torch.max(result, 1)
        for vec, pred in zip(test_vec, predicted):
            max_index = vec.index(max(vec))
            print(f"输入：{vec}, 真实最大位置：{max_index}, 预测类别：{pred.item()}")
