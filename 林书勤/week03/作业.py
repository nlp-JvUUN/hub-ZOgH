# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 可选，用于可视化

"""
基于pytorch的文本多分类任务模型比较框架
输入一个固定长度为5的文本，根据字符‘a’的出现情况进行分类：
- 0个‘a’：第0类
- 1个‘a’：出现在第几位就是第几类（1-5）
- 2个‘a’：第6类
- 3个‘a’：第7类
- 4个‘a’：第8类
- 5个‘a’：第9类
共计10个类别（0-9）
"""

# ==================== 1. 数据构建模块====================
def build_vocab():
    """构建字符到索引的词典"""
    chars = "abcdefghijk王1"  # 扩充了字符集，使数据更多样
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length=5):
    """生成一个样本及其标签"""
    # 随机生成文本
    x = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]  # 排除‘pad’和‘unk’
    
    # 根据规则生成标签
    count = x.count('a')
    if count == 0:
        y = 0
    elif count == 1:
        y = x.index('a') + 1
    else:  # count >= 2
        y = 5 + count  # 5(文本长度) + 超过1个的a的数量
        y = min(y, 9)  # 确保类别不超过9（当5个字符全是‘a’时）
    
    # 将字符转换为索引
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length=5):
    """构建数据集"""
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 注意：这里直接存储y，而非[y]，以便后续处理
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# ==================== 2. 模型定义模块  ====================
class SequenceClassificationModel(nn.Module):
    """一个通用的序列分类模型，可通过参数指定RNN类型"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, rnn_type='lstm', num_layers=1):
        super(SequenceClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 根据参数选择不同的循环神经网络层
        rnn_type = rnn_type.lower()
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose from 'rnn', 'lstm', 'gru'")
        
        # 分类层
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        rnn_out, _ = self.rnn(x)  # _ 代表最终的隐藏状态，这里我们不需要
        # 取最后一个时间步的输出作为整个序列的表示
        last_hidden_state = rnn_out[:, -1, :]  # (batch, hidden_size)
        logits = self.classifier(last_hidden_state)  # (batch, num_classes)
        
        if y is not None:
            return self.loss(logits, y)
        else:
            return torch.softmax(logits, dim=-1)

# ==================== 3. 训练与评估模块 ====================
def evaluate(model, vocab, sentence_length, test_num=200):
    """评估模型在测试集上的准确率"""
    model.eval()
    x, y_true = build_dataset(test_num, vocab, sentence_length)
    correct = 0
    with torch.no_grad():
        y_pred_prob = model(x)
        y_pred = torch.argmax(y_pred_prob, dim=-1)
        correct = (y_pred == y_true).sum().item()
    accuracy = correct / test_num
    print(f"评估结果：正确数 {correct}/{test_num}, 准确率 {accuracy:.4f}")
    return accuracy

def train_model(model, vocab, config, model_name="Model"):
    """训练一个模型的通用流程"""
    print(f"\n开始训练 {model_name}...")
    # 准备数据
    train_x, train_y = build_dataset(config['train_sample'], vocab, config['sentence_length'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_history = []
    acc_history = []
    
    for epoch in range(config['epoch_num']):
        model.train()
        epoch_losses = []
        
        # 迷你批次训练
        for batch_idx in range(0, config['train_sample'], config['batch_size']):
            x_batch = train_x[batch_idx: batch_idx + config['batch_size']]
            y_batch = train_y[batch_idx: batch_idx + config['batch_size']]
            
            loss = model(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            # 可选：梯度裁剪，防止梯度爆炸，对RNN尤其有用
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        # 每轮结束后评估
        acc = evaluate(model, vocab, config['sentence_length'], test_num=200)
        acc_history.append(acc)
        
        print(f"Epoch {epoch+1:3d}/{config['epoch_num']} | 平均Loss: {avg_loss:.4f} | 准确率: {acc:.4f}")
    
    return loss_history, acc_history

# ==================== 4. 主函数：比较不同模型 ====================
def main():
    # 实验配置
    config = {
        'epoch_num': 15,
        'batch_size': 32,
        'train_sample': 2000,  # 训练样本数
        'sentence_length': 5,
        'embedding_dim': 10,
        'hidden_size': 16,
        'num_classes': 10,
        'learning_rate': 0.001,
    }
    
    vocab = build_vocab()
    vocab_size = len(vocab)
    
    # 定义要比较的模型
    models_to_train = {
        'RNN': 'rnn',
        'LSTM': 'lstm',
        'GRU': 'gru'
    }
    
    results = {}
    
    # 依次训练并比较每个模型
    for name, rnn_type in models_to_train.items():
        # 初始化模型
        model = SequenceClassificationModel(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_classes=config['num_classes'],
            rnn_type=rnn_type,
            num_layers=1  # 可以使用多层，这里设为1保持简单
        )
        # 训练
        loss_hist, acc_hist = train_model(model, vocab, config, model_name=name)
        results[name] = {'loss': loss_hist, 'acc': acc_hist, 'model': model}
        
        # 保存模型
        torch.save(model.state_dict(), f"{name}_model.pth")
        print(f"{name} 模型已保存为 {name}_model.pth")
    
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("词表已保存为 vocab.json")
    
    # ==================== 5. 结果可视化 ====================
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['loss'], label=f'{name} Loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss of Different RNN Cells')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res['acc'], label=f'{name} Acc', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy of Different RNN Cells')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
    
    # 打印最终性能比较
    print("\n" + "="*50)
    print("模型最终性能比较：")
    for name, res in results.items():
        final_acc = res['acc'][-1]
        print(f"{name:4s} : 最终准确率 = {final_acc:.4f}")
    print("="*50)

# ==================== 6. 预测函数 ====================
def predict(model_path, vocab_path, input_strings, rnn_type='lstm'):
    """使用训练好的模型进行预测"""
    # 加载词表
    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab = json.load(f)
    
    # 模型参数
    config = {
        'embedding_dim': 10,
        'hidden_size': 16,
        'num_classes': 10,
        'sentence_length': 5,
    }
    
    # 重建模型结构
    model = SequenceClassificationModel(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
        rnn_type=rnn_type
    )
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 准备输入数据
    sen_len = config['sentence_length']
    processed_inputs = []
    for s in input_strings:
        # 处理输入：填充或截断
        s = s[:sen_len].ljust(sen_len, ' ')  # 右填充空格
        indices = [vocab.get(char, vocab['unk']) for char in s]
        processed_inputs.append(indices)
    
    # 预测
    with torch.no_grad():
        inputs_tensor = torch.LongTensor(processed_inputs)
        outputs = model(inputs_tensor)
        predictions = torch.argmax(outputs, dim=-1)
    
    # 打印结果
    print(f"\n使用 {rnn_type.upper()} 模型进行预测：")
    for i, (inp_str, pred) in enumerate(zip(input_strings, predictions)):
        print(f"  输入: “{inp_str}” -> 预测类别: {pred.item()}")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()
    
    # 示例：使用训练好的LSTM模型进行预测
    test_strings = ["abcde", "aacde", "aaade", "aaaaa", "x王yz1", "bacde"]
    predict("LSTM_model.pth", "vocab.json", test_strings, rnn_type='lstm')
