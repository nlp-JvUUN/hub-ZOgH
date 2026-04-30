import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
# 设置matplotlib使用默认后端
import matplotlib

matplotlib.use('TkAgg')  # 使用TkAgg后端，或者尝试 'Qt5Agg'


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        # self.linear2 = nn.Linear(input_size, input_size)
        # self.linear3 = nn.Linear(input_size, input_size)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    def forward(self, x, y):
        y_pred = self.activation(self.linear1(x))
        # y_pred = self.activation(self.linear2(y_pred))
        # y_pred = self.activation(self.linear3(y_pred))

        return self.loss(y_pred, y)  # 预测值和真实值计算损失

    def predict(self, x):
        y_pred = self.activation(self.linear1(x))
        # y_pred = self.activation(self.linear2(y_pred))
        # y_pred = self.activation(self.linear3(y_pred))

        return y_pred


# 生成模型数据
def build_model_data(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x = random.sample(range(1, 6), 5)
        y = [1 if num == max(x) else 0 for num in x]
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, test_sample_num=100):
    model.eval()
    test_X, test_Y = build_model_data(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.predict(test_X)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, test_Y):  # 与真实标签进行对比
            # 转化预测结果，把最大值位置设置为1，其他位置设置为0
            y_p = [1. if num == max(y_p) else 0. for num in y_p]
            # 找到预测概率最大的位置
            pred_max_index = np.argmax(y_p)
            true_max_index = np.argmax(y_t)
            # print(f"预测值：{y_p}")
            # print(f"实际值：{y_t}")

            # 判断是否相等
            if pred_max_index == true_max_index:
                correct += 1
            else:
                wrong += 1

    # 计算准确率
    accuracy = correct / (correct + wrong)
    print(f"\n=== 测试结果 ===")
    print(f"正确: {correct}")
    print(f"错误: {wrong}")
    print(f"准确率: {accuracy:.2%}")
    print("=" * 30)

    return accuracy


def load_model(model_path="model.bin"):
    """加载训练好的模型"""
    try:
        # 加载保存的模型参数
        checkpoint = torch.load(model_path)

        # 检查保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 新格式：包含模型状态和元数据
            input_size = checkpoint['input_size']
            model_state_dict = checkpoint['model_state_dict']
        else:
            # 旧格式：只包含模型状态
            # 需要从模型结构推断input_size
            # 这里假设输入大小为5，因为你的模型是固定的
            input_size = 5
            model_state_dict = checkpoint

        # 创建模型实例
        model = TorchModel(input_size)

        # 加载模型参数
        model.load_state_dict(model_state_dict)
        model.eval()  # 设置为评估模式

        print(f"模型已从 '{model_path}' 加载")
        return model
    except:
        return None


def predict_batch_input(test_sample_num):
    """使用训练好的模型进行批量预测"""
    model = load_model("model.bin")
    if model is None:
        print("无法加载模型，请先训练模型")
        return

    acc = evaluate(model, test_sample_num)
    print(f"输入样本{test_sample_num}个，准确率为{acc * 100:.2f}%")


def save_model(model, model_path="model.bin"):
    """保存模型（包含元数据）"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': 5,  # 你的模型输入大小固定为5
        'model_type': 'TorchModel'
    }
    torch.save(checkpoint, model_path)
    print(f"模型已保存为 '{model_path}'")


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_model_data(train_sample)
    print(train_x)
    print(train_y)
    # 训练过程
    model.train()
    for epoch in range(epoch_num):
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 如果成功率大于99%,结束训练
        # if acc >= 0.98:
        #     # 准确率达到要求，结束训练
        #     print(f"当前准确率为{acc * 100}%，准确率达到要求，结束训练")
        #     break
    # 保存模型
    save_model(model, "model.bin")
    # 画图
    # 准确率曲线
    plt.subplot(1, 2, 1)
    epochs = list(range(1, len(log) + 1))
    accuracies = [l[0] for l in log]
    plt.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True, alpha=0.3)

    # 损失曲线
    plt.subplot(1, 2, 2)
    losses = [l[1] for l in log]
    plt.plot(epochs, losses, 'r-s', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
    print("\n图表已保存为 'training_results.png'")

    # 如果仍然无法显示，可以注释掉show()，只保存图片
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图表: {e}")
        print("图表已保存为 'training_results.png'，请在文件管理器中查看")


if __name__ == "__main__":
    # main()
    predict_batch_input(1000)
