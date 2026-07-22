# RNN/LSTM 文本多分类实验

任务：输入任意 5 个汉字且恰好包含一个“你”，预测“你”出现在第几位，类别为 1 到 5。

数据：脚本随机合成训练集和测试集；非“你”的位置从固定汉字表中随机采样。

模型：纯 Python 标准库实现字符 embedding、vanilla RNN、LSTM、softmax 多分类和 BPTT。

最终结果：

- RNN: epoch=15, loss=0.0005, train_acc=1.000, test_acc=1.000
- LSTM: epoch=15, loss=0.0028, train_acc=1.000, test_acc=1.000

运行方式：

```bash
python outputs/train_text_rnn_lstm.py
```
