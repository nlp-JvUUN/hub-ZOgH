# 第二周作业：
"""
五分类模型训练示例
基于PyTorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，其中哪一维数字最大，就为第几类（0-4类）
"""

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


class TorchModel5Class(nn.Module):
    """五分类神经网络模型"""
    
    def __init__(self, input_size, num_classes=5):
        super(TorchModel5Class, self).__init__()
        # 线性层：5维输入 -> 5维输出（对应5个类别）
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失（内置了softmax）
    
    def forward(self, x, y=None):
        """
        前向传播
        x: 输入特征，形状 (batch_size, 5)
        y: 真实标签，形状 (batch_size,)，取值范围 0-4
        """
        # 线性变换得到logits
        logits = self.linear(x)  # (batch_size, 5)
        
        if y is not None:
            # 训练模式：计算交叉熵损失
            # CrossEntropyLoss期望logits和类别索引（不是one-hot）
            return self.loss(logits, y)
        else:
            # 预测模式：返回softmax概率
            return torch.softmax(logits, dim=1)


def build_sample():
    """
    生成一个样本
    返回：5维随机向量，标签为最大值所在的索引（0-4）
    """
    x = np.random.random(5)  # 生成5个0-1之间的随机数
    # 找到最大值的索引作为类别标签
    label = np.argmax(x)  # 0, 1, 2, 3, 4
    return x, label


def build_dataset(total_sample_num):
    """
    生成训练数据集
    total_sample_num: 样本总数
    返回：特征张量和标签张量
    """
    X = []
    Y = []
    
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  
    
    # 转换为PyTorch张量
    # 注意：分类标签需要使用LongTensor
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    """
    评估模型准确率
    使用100个测试样本计算准确率
    """
    model.eval()  # 切换到评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        # 前向传播得到预测概率
        y_pred_probs = model(x)  # 形状: (100, 5)
        
        # 获取预测类别（概率最大的类别）
        y_pred_classes = torch.argmax(y_pred_probs, dim=1)  # 形状: (100,)
        
        # 计算正确预测的数量
        correct = (y_pred_classes == y).sum().item()
        total = y.size(0)
    
    accuracy = correct / total
    
    # 统计每个类别的分布
    class_counts = torch.bincount(y, minlength=5)
    print(f"测试集类别分布: {dict(enumerate(class_counts.tolist()))}")
    print(f"正确预测个数: {correct}/{total}, 准确率: {accuracy:.4f}")
    
    return accuracy


def main():
    """主训练函数"""
    
    # 超参数配置
    epoch_num = 30          # 训练轮数
    batch_size = 32         # 批量大小
    train_sample = 5000     # 训练样本总数
    input_size = 5          # 输入维度
    num_classes = 5         # 类别数量
    learning_rate = 0.01    # 学习率
    
    print("=" * 50)
    print("五分类模型训练配置:")
    print(f"  训练轮数: {epoch_num}")
    print(f"  批量大小: {batch_size}")
    print(f"  训练样本: {train_sample}")
    print(f"  学习率: {learning_rate}")
    print("=" * 50)
    
    # 创建模型
    model = TorchModel5Class(input_size, num_classes)
    
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练日志
    log = []
    
    # 生成训练数据
    print("生成训练数据...")
    train_x, train_y = build_dataset(train_sample)
    
    # 统计训练集类别分布
    train_class_counts = torch.bincount(train_y, minlength=5)
    print(f"训练集类别分布: {dict(enumerate(train_class_counts.tolist()))}")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式
        epoch_loss = []
        
        # 计算批次数
        num_batches = train_sample // batch_size
        
        for batch_idx in range(num_batches):
            # 获取当前批次的索引范围
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # 提取批次数据
            batch_x = train_x[start_idx:end_idx]
            batch_y = train_y[start_idx:end_idx]
            
            # 训练四步曲
            # 1. 前向传播：计算损失
            loss = model(batch_x, batch_y)
            
            # 2. 反向传播：计算梯度
            loss.backward()
            
            # 3. 参数更新
            optimizer.step()
            
            # 4. 梯度清零
            optimizer.zero_grad()
            
            # 记录损失
            epoch_loss.append(loss.item())
        
        # 计算本轮平均损失
        avg_loss = np.mean(epoch_loss)
        
        # 每轮结束后评估
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5轮评估一次，第一轮也评估
            print(f"\n第 {epoch+1:2d}/{epoch_num} 轮 | 平均损失: {avg_loss:.6f}")
            acc = evaluate(model)
            log.append([epoch + 1, acc, avg_loss])
        else:
            print(f"第 {epoch+1:2d}/{epoch_num} 轮 | 平均损失: {avg_loss:.6f}")
            # 即使不评估也记录损失
            log.append([epoch + 1, log[-1][1] if log else 0, avg_loss])
    
    # 训练完成
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最终准确率: {log[-1][1]:.4f}")
    print("=" * 50)
    
    # 保存模型
    model_path = "model_5class.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    epochs = [item[0] for item in log]
    accs = [item[1] for item in log]
    plt.plot(epochs, accs, 'b-o', label='准确率', linewidth=2, markersize=4)
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')
    plt.title('准确率变化曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim([0, 1.05])
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    losses = [item[2] for item in log]
    plt.plot(epochs, losses, 'r-s', label='损失', linewidth=2, markersize=4)
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title('损失变化曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_5class.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model


def predict(model_path, input_vecs):
    """
    使用训练好的模型进行预测
    model_path: 模型文件路径
    input_vecs: 输入向量列表，每个向量长度为5
    """
    # 创建模型实例
    model = TorchModel5Class(input_size=5, num_classes=5)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()  # 切换到评估模式
    
    with torch.no_grad():  # 不计算梯度
        # 转换为张量
        input_tensor = torch.FloatTensor(input_vecs)
        
        # 预测
        probs = model(input_tensor)  # 形状: (n_samples, 5)
        pred_classes = torch.argmax(probs, dim=1)  # 预测类别
    
    # 打印预测结果
    print("\n" + "=" * 60)
    print("预测结果详情:")
    print("=" * 60)
    print(f"{'输入向量':<40} {'预测类别':<8} {'各类别概率'}")
    print("-" * 60)
    
    for i, (vec, pred, prob) in enumerate(zip(input_vecs, pred_classes, probs)):
        vec_str = np.array2string(np.array(vec), precision=3, suppress_small=True)
        prob_str = ', '.join([f"{p:.3f}" for p in prob])
        print(f"{vec_str:<40} 第{pred}类    [{prob_str}]")
    
    print("=" * 60)
    return pred_classes.tolist()


def demo_prediction():
    """演示预测功能"""
    print("\n" + "=" * 50)
    print("模型预测演示")
    print("=" * 50)
    
    # 生成一些测试样本
    test_samples = []
    
    # 手动创建一些有明显特征的样本
    test_samples.append([0.9, 0.1, 0.2, 0.3, 0.4])  # 最大在第0维
    test_samples.append([0.1, 0.9, 0.2, 0.3, 0.4])  # 最大在第1维
    test_samples.append([0.1, 0.2, 0.9, 0.3, 0.4])  # 最大在第2维
    test_samples.append([0.1, 0.2, 0.3, 0.9, 0.4])  # 最大在第3维
    test_samples.append([0.1, 0.2, 0.3, 0.4, 0.9])  # 最大在第4维
    
    # 再随机生成5个样本
    for _ in range(5):
        test_samples.append(np.random.random(5).tolist())
    
    # 预测
    try:
        predictions = predict("model_5class.pth", test_samples)
        print(f"\n预测完成！共预测 {len(predictions)} 个样本")
    except FileNotFoundError:
        print("\n提示: 请先运行main()函数训练并保存模型")


# 使用训练好的模型做预测示例
if __name__ == "__main__":
    # 训练模型
    trained_model = main()
    
    # 演示预测
    demo_prediction()
