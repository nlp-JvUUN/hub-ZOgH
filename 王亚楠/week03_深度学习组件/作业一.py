import torch
import torch.nn as nn
import numpy as np
import random
import json

# 1. 模型
class TorchModel(nn.Module):
  def __init__(self, vector_dim, sentence_length, vocab):
    super(TorchModel, self).__init__()
    self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx = 0)
    self.RNN = nn.RNN(vector_dim, hidden_size = 128, batch_first = True)
    self.hidden_size = 128
    self.classify = nn.Linear(128, sentence_length + 2)
    self.loss = nn.CrossEntropyLoss()
  def forward(self,x, y = None):
    x = self.embedding(x)
    rnn_out, hidden = self.RNN(x)
    print(f"rnn_out.shape:{rnn_out.shape}")
    print(f"hidden:{hidden}")
    x = rnn_out[:, -1, :]
    y_pred = self.classify(x)
    if y is not None:
      return self.loss(y_pred, y)
    else:
      y_pred = torch.softmax(y_pred, dim = -1)
      return y_pred
# 2. 数据
def build_vocab():
  chars = "你我他二三abcdfrgk"
  vocab = {"pad":0}
  for index, char in enumerate(chars):
    vocab[char] = index + 1
  vocab['unk'] = len(vocab)
  return vocab
def build_sample(vocab, sentence_length):
  x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
  if '我' in x:
    if x.count('你') == 1:
      y = x.index('你')
    else:
      y = sentence_length
  else:
    y = sentence_length + 1
  x = [vocab.get(word, vocab['unk']) for word in x]
  return x, y

# 3. 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
  dataset_x = []
  dataset_y = []
  for i in range(sample_length):
    x, y = build_sample(vocab, sentence_length)
    dataset_x.append(x)
    dataset_y.append(y)
  return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 4. 建立模型
def build_model(vocab, char_dim, sentence_length):
  model = TorchModel(char_dim, sentence_length, vocab)
  return model

# 5. 验证
def evaluate(model, vocab, sample_length):
  model.eval()
  x, y = build_dataset(200, vocab, sample_length)
  correct, wrong = 0, 0
  with torch.no_grad():
    y_pred = model(x)
    preds = torch.argmax(y_pred, dim = -1)
    correct = (preds == y).sum().item()
    wrong = len(y) - correct
  print('正确预测个数：%d, 正确率： %f' % (correct, correct / (correct + wrong)))
  return correct / (correct + wrong)

# 6. 主程序
def main():
  epoch_num = 10
  batch_size = 20
  train_sample = 500
  char_dim = 20
  sentence_length = 6
  learning_rate = 0.005
  vocab = build_vocab()
  model = build_model(vocab, char_dim, sentence_length)
  optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
  log = []
  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for batch in range(int(train_sample / batch_size)):
      x, y = build_dataset(batch_size, vocab, sentence_length)
      optim.zero_grad()
      loss = model(x, y)
      loss.backward()
      optim.step()
      watch_loss.append(loss.item())
    print("==========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    acc = evaluate(model, vocab, sentence_length)
    log.append([acc, np.mean(watch_loss)])
  torch.save(model.state_dict(), "week3model.pth")
  writer = open("vocab.json", "w", encoding = "utf8")
  writer.write(json.dumps(vocab, ensure_ascii = False, indent = 2))
  writer.close()
  return

# 7. 预测
def predict(model_path, vocab_path, input_strings):
  char_dim = 20
  sentence_length = 6
  vocab = json.load(open(vocab_path, "r", encoding = "utf8"))
  model = build_model(vocab, char_dim, sentence_length)
  model.load_state_dict(torch.load(model_path))
  x = []
  for input_string in input_strings:
    x.append([vocab[char] for char in input_string])
  model.eval()
  with torch.no_grad():
    result = model.forward(torch.LongTensor(x))
  for i, input_string in enumerate(input_string):
    prob, class_index = torch.max(result[i], dim = -1)
    print("输入：%s, 预测类别索引：%d, 置信度：%f" % (input_string, class_index.item(), prob.item()))
if __name__ == "__main__":
  main()
