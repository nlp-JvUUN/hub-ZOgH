import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from torch import nn

"""
基于pytorch框架编写模型训练
完成一个多分类任务的训练，一个随机向量，哪一维数字最大就属于第几类
"""

# 1. 数据
def build_sample_data():
  x = np.random.random(5)
  y = np.argmax(x)
  return x, y

def build_dataset(total_sample_num):
  X, Y = [], []
  for i in range(total_sample_num):
    x, y = build_sample_data()
    X.append(x)
    Y.append(y)
  return torch.FloatTensor(X), torch.LongTensor(Y)

# 2. 模型
class TorchModel(nn.Module):
  def __init__(self, input_size):
    super(TorchModel, self).__init__()
    self.linear = nn.Linear(input_size, input_size)
    self.loss = nn.CrossEntropyLoss()
  def forward(self, x, y = None):
    y_pred = self.linear(x)
    if y is not None:
      return self.loss(y_pred, y)
    else:
      y_pred = torch.softmax(y_pred, dim = -1)
      return y_pred

# 3. 测试 evaluate
def evaluate(model):
  model.eval()
  test_sample_num = 1000
  x, y = build_dataset(test_sample_num)
  correct, wrong = 0, 0
  with torch.no_grad():
    y_pred = model(x)
    for y_p, y_t in zip(y_pred, y):
      if np.argmax(y_p) == y_t:
        correct += 1
      else:
        wrong += 1
  print("正确的预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
  return correct / (correct + wrong)

# 4. 训练代码
def train(epoch_num, batch_size, train_sample_num, learning_rate, model):
  optimize = torch.optim.Adam(model.parameters(), lr = learning_rate)
  log = []
  train_x, train_y = build_dataset(train_sample_num)
  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for batch_index in range(train_sample_num // batch_size):
      x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
      y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
      loss = model(x,y)
      loss.backward()
      optimize.step()
      optimize.zero_grad()
      watch_loss.append(loss.item())
    print('===============\n第%d轮平均loss为%f' % (epoch + 1, np.mean(watch_loss)))
    acc = evaluate(model)
    log.append([acc, float(np.mean(watch_loss))])
  torch.save(model.state_dict(), 'modelCrossEntropy.pt')
  return log

# 5. 主程序
def main():
  epoch_num = 100
  batch_size = 5
  train_sample_num = 1000
  input_size = 5
  learning_rate = 0.01
  model = TorchModel(input_size)
  log = train(epoch_num, batch_size, train_sample_num, learning_rate, model)
  print(log)
  plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
  plt.plot(range(len(log)), [l[1] for l in log], label = 'loss')
  plt.legend()
  plt.show()

# 6. 预测
def predict(model_path, input_vec):
  input_size = 5
  model = TorchModel(input_size)
  model.load_state_dict(torch.load(model_path))
  print(model.state_dict())
  model.eval()
  with torch.no_grad():
    result = model.forward(torch.FloatTensor(input_vec))
  for vec, res in zip(input_vec, result):
    print('输入：%s, 预测最大值位置：%d， 预测概率值%f' % (vec, np.argmax(res), res[np.argmax(res)]))

if __name__ == '__main__':
  main()
