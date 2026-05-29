import os
import matplotlib
matplotlib.use('TkAgg')  # Windows系统首选，mac/linux可试'QtAgg'/'GTK3Agg'
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np

"""
多分类任务：
输入：5维随机向量
规则：哪一维数字最大 → 属于第几类（共5类：0,1,2,3,4）
模型：线性层 + Softmax + 交叉熵损失
"""


# ===================== 1. 定义多分类模型 =====================
class TorchMultiClassModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchMultiClassModel, self).__init__()
        # 线性层：输入维度 → 类别数
        self.linear = nn.Linear(input_size, class_num)
        # 多分类必须用 CrossEntropyLoss（内部自带Softmax）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 前向传播
        logits = self.linear(x)  # (batch, 5) → (batch, 5类)

        # 有标签 → 算loss
        if y is not None:
            return self.loss(logits, y)
        # 无标签 → 返回预测类别
        else:
            # softmax 转概率
            predict_prob = torch.softmax(logits, dim=1)
            # 取概率最大的下标 = 预测类别
            predict_class = torch.argmax(predict_prob, dim=1)
            return predict_class, predict_prob


# ===================== 2. 生成样本（核心规则） =====================
def build_sample(input_size=5):
    """
    生成1个样本：5维随机向量
    哪一维最大 → 标签就是几
    """
    x = np.random.random(input_size)  # 生成 0~1 随机数
    y = np.argmax(x)  # 最大值所在下标 = 标签
    return x, y


def build_dataset(total_num, input_size=5):
    X = []
    Y = []
    for _ in range(total_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    # 转成tensor
    X = torch.FloatTensor(np.array(X))  # 先转成单个ndarray，再转张量
    Y = torch.LongTensor(np.array(Y)) # 分类标签必须是 LongTensor
    return X, Y


# ===================== 3. 评估准确率 =====================
def evaluate(model, input_size=5):
    model.eval()
    test_x, test_y = build_dataset(200, input_size)
    correct = 0
    with torch.no_grad():
        pred_class, _ = model(test_x)
        correct = (pred_class == test_y).sum().item()
    acc = correct / len(test_y)
    print(f"正确：{correct}，总样本：{len(test_y)}，准确率：{acc:.4f}")
    return acc


# ===================== 4. 训练主函数 =====================
def main():
    # 超参数
    input_size = 5  # 输入5维向量
    class_num = 5  # 5个分类
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 批次大小
    train_sample = 5000  # 总样本
    lr = 0.01  # 学习率

    # 模型 + 优化器
    model = TorchMultiClassModel(input_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    # 构建训练集
    train_x, train_y = build_dataset(train_sample, input_size)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        losses = []
        for i in range(train_sample // batch_size):
            # 取一个batch
            x = train_x[i * batch_size: (i + 1) * batch_size]
            y = train_y[i * batch_size: (i + 1) * batch_size]

            # 前向传播 + 算loss
            loss = model(x, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()
            # 清空梯度
            optim.zero_grad()

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"\n===== 第 {epoch + 1} 轮，平均loss：{avg_loss:.4f} =====")
        acc = evaluate(model)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.pth")
    print("\n模型已保存：multi_class_model.pth")

    # 画图
    plt.figure(figsize=(8, 4))
    plt.plot([l[0] for l in log], label="acc", color='blue')
    plt.plot([l[1] for l in log], label="loss", color='red')
    plt.legend()
    plt.title("多分类训练曲线")
    plt.show()


# ===================== 5. 预测函数 =====================
def predict(model_path, input_vecs):
    input_size = 5
    class_num = 5
    model = TorchMultiClassModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(input_vecs)
        pred_class, pred_prob = model(x)

    # 打印结果
    for i, vec in enumerate(input_vecs):
        cls = pred_class[i].item()
        prob = pred_prob[i][cls].item()
        print(f"输入向量：{vec}")
        print(f"预测类别：{cls}，置信度：{prob:.4f}\n")


# ===================== 运行 =====================
if __name__ == "__main__":
    # 训练
    main()

    # 训练完后预测测试
    test_vecs = [
        [0.1, 0.2, 0.5, 0.1, 0.1],  # 第2维最大 → 类别2
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 第0维最大 → 类别0
        [0.2, 0.7, 0.1, 0.1, 0.1],  # 第1维最大 → 类别1
        [0.1, 0.1, 0.1, 0.8, 0.1],  # 第3维最大 → 类别3
        [0.1, 0.1, 0.1, 0.1, 0.9],  # 第4维最大 → 类别4
    ]
    predict("multi_class_model.pth", test_vecs)
