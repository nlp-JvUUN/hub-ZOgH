#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五字文本中定位字符 '你' 的位置分类任务
=================================================
任务描述:
    给定一个任意包含"你"字的五个字的文本，"你"在第几位，就属于第几类。
    例如: "你abc" -> 第1类, "a你bc" -> 第2类, "ab你c" -> 第3类, "abc你" -> 第4类, "abcy你" -> 第5类

数据生成方式:
    每个样本在一个固定字母表(a-z)中，随机选一个位置放入"你"，其余位填入a-z。
    这样生成五字序列，并记录"你"字所在的索引位置作为类别标签(0-4)。

模型选择:
    支持 RNN 和 LSTM 两种循环神经网络模型进行序列分类。
    模型结构: Embedding -> RNN/LSTM -> 全连接层 -> 5类输出

使用方法:
    python 作业3.py --model lstm --epochs 20
    参数说明:
        --model: 选择'rnn'或'lstm'模型
        --train_size: 训练样本数量
        --test_size: 测试样本数量
        --epochs: 训练轮数
        --embed_dim: 词嵌入维度
        --hidden_dim: 隐藏层维度
        --batch: 批次大小
        --seed: 随机种子
        --save: 模型保存路径
"""

# ============================================================
# 导入所需的模块
# ============================================================

import argparse  # 用于解析命令行参数
import json      # 用于JSON序列化(虽然本例未直接使用,但保留以备扩展)
import os        # 用于操作系统功能(如路径处理)
import random    # 用于生成随机数据
from typing import List  # 用于类型注解,声明List类型

import torch                   # PyTorch深度学习框架核心
import torch.nn as nn         # PyTorch神经网络模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具


# ============================================================
# 数据集类定义
# ============================================================

class CharPosDataset(Dataset):
    """
    五字文本位置分类数据集

    该数据集用于生成包含"你"字的五字序列,并标注"你"字在序列中的位置(0-4)。
    位置0表示"你"在第1位,位置1表示"你"在第2位,以此类推。

    属性:
        n: 数据集样本数量
        seed: 随机种子,确保数据可复现
        alphabet: 可用的字符集(默认为a-z小写字母)
        special: 特殊字符"你",这是我们要定位的目标字符
        vocab: 词表,包含所有可用字符
        char2idx: 字符到索引的映射字典
    """

    def __init__(self, n_samples: int = 10000, seed: int = 0, alphabet: List[str] = None):
        """
        初始化数据集

        参数:
            n_samples: 要生成的样本数量,默认10000
            seed: 随机种子,默认0,用于确保数据可复现
            alphabet: 字符列表,默认None时使用26个英文字母
        """
        # 调用父类Dataset的初始化方法
        super().__init__()

        # 保存样本数量和随机种子
        self.n = n_samples
        self.seed = seed

        # 如果没有提供alphabet,则使用26个小写英文字母作为默认字符集
        if alphabet is None:
            alphabet = list("abcdefghijklmnopqrstuvwxyz")

        # 保存字母表
        self.alphabet = alphabet

        # 定义特殊字符"你",这是我们要定位的目标字符
        self.special = '你'

        # 构建词表: 将字母表和特殊字符"你"合并
        self.vocab = self.alphabet + [self.special]

        # 构建字符到索引的映射字典,如{'a':0, 'b':1, ..., 'z':25, '你':26}
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}

        # 创建随机数生成器,使用指定种子确保可复现性
        self.rng = random.Random(seed)

        # 记录数据集长度(用于__len__方法返回)
        self._len = self.n

    def __len__(self):
        """
        返回数据集的样本数量

        返回:
            int: 数据集中包含的样本总数
        """
        return self._len

    def __getitem__(self, idx):
        """
        获取指定索引的样本

        该方法随机生成一个五字序列,其中包含一个"你"字,
        并返回该序列的字符索引以及"你"字的位置标签。

        参数:
            idx: 样本索引(本实现中未使用,因为是随机生成的)

        返回:
            tuple: (序列索引张量, 位置标签张量)
                - 序列索引张量: 形状为(5,)的LongTensor,每个元素是字符的索引
                - 位置标签张量: 形状为()的LongTensor,值为0-4,表示"你"字的位置
        """
        # 随机选择"你"字在序列中的位置,范围为0-4
        # 0表示第1位,1表示第2位,...,4表示第5位
        pos = self.rng.randrange(0, 5)  # 0..4

        # 生成一个长度为5的随机序列,使用字母表中的随机字符填充
        seq = [self.rng.choice(self.alphabet) for _ in range(5)]

        # 将随机选择的pos位置替换为特殊字符"你"
        seq[pos] = self.special

        # 将字符序列转换为对应的索引序列
        # 例如: ['a', '你', 'c', 'd', 'e'] -> [0, 26, 2, 3, 4]
        indices = [self.char2idx[c] for c in seq]

        # 返回两个张量:
        # - 第一个是输入序列的索引,形状为(5,),类型为long
        # - 第二个是"你"字的位置标签,形状为(),类型为long
        return torch.tensor(indices, dtype=torch.long), torch.tensor(pos, dtype=torch.long)


# ============================================================
# 模型类定义
# ============================================================

class CharPosModel(nn.Module):
    """
    基于RNN/LSTM的五字文本位置分类模型

    模型结构:
        1. 嵌入层(Embedding): 将字符索引转换为密集向量表示
        2. 循环神经网络层(RNN/LSTM): 捕捉序列中的时序信息
        3. 全连接层(Linear): 将隐藏状态映射到5类输出

    前向传播流程:
        输入(batch_size, 5) -> 嵌入(batch_size, 5, embed_dim) ->
        RNN/LSTM处理 -> 取最后一个时刻的隐藏状态 ->
        全连接层 -> 输出(batch_size, 5)
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, model_type: str = 'lstm'):
        """
        初始化模型

        参数:
            vocab_size: 词表大小,即不同字符的总数
            embed_dim: 词嵌入维度,将每个字符映射为embed_dim维的向量
            hidden_dim: RNN/LSTM隐藏层的维度
            model_type: 模型类型,'lstm'或'rnn',默认'lstm'
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 保存模型类型,供前向传播时选择不同的处理方式
        self.model_type = model_type

        # 创建嵌入层,将vocab_size个字符映射为embed_dim维的向量
        # 输入: (batch_size, seq_len) 索引序列
        # 输出: (batch_size, seq_len, embed_dim) 密集向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 根据model_type选择创建LSTM或RNN层
        # batch_first=True表示输入输出形状的第一个维度是batch_size
        if model_type == 'lstm':
            # 创建LSTM层
            # 输入维度: embed_dim, 输出维度: hidden_dim
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            # 创建RNN层
            # 输入维度: embed_dim, 输出维度: hidden_dim
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        # 创建全连接层,将隐藏状态映射到5类输出
        # 输入维度: hidden_dim, 输出维度: 5(五个位置类别)
        # 5个类别分别对应"你"在第1、2、3、4、5位
        self.fc = nn.Linear(hidden_dim, 5)  # 5 classes, positions 0..4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量,形状为(batch_size, 5),包含字符索引

        返回:
            torch.Tensor: 输出张量,形状为(batch_size, 5),表示各类别的logits
        """
        # x的形状为 [B, 5],其中B是batch_size

        # 通过嵌入层将字符索引转换为密集向量
        # 输入: [B, 5] -> 输出: [B, 5, E],E是embed_dim
        emb = self.embedding(x)  # [B, 5, E]

        # 根据模型类型选择不同的处理方式
        if self.model_type == 'lstm':
            # LSTM有两个输出:所有时刻的输出和最后时刻的隐藏状态(c, h)
            # out: [B, 5, H], hn: [1, B, H], cn: [1, B, H]
            out, (hn, cn) = self.rnn(emb)

            # 取最后一个隐藏状态hn[-1],形状为[B, H]
            # 对于LSTM,hn[-1]是最后时刻的隐藏状态
            last = hn[-1]  # [B, H]
        else:
            # RNN的输出处理方式
            # out: [B, 5, H], hidden: [1, B, H]
            out, hidden = self.rnn(emb)

            # 取最后一个时刻的输出,形状为[B, H]
            # out[:, -1, :]获取序列最后一个时刻的所有隐藏状态
            last = out[:, -1, :]  # [B, H]

        # 通过全连接层将隐藏状态映射到5类输出
        # 输入: [B, H] -> 输出: [B, 5]
        logits = self.fc(last)  # [B, 5]

        # 返回logits,用于计算交叉熵损失
        return logits


# ============================================================
# 训练函数
# ============================================================

def train(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 10, lr: float = 0.001):
    """
    训练模型

    参数:
        model: 要训练的神经网络模型
        train_loader: 训练数据加载器
        device: 计算设备(CPU或GPU)
        epochs: 训练轮数,默认10
        lr: 学习率,默认0.001

    训练流程:
        1. 设置损失函数为交叉熵损失
        2. 设置优化器为Adam
        3. 循环多个epoch,每个epoch遍历所有训练数据
        4. 每个batch: 前向传播 -> 计算损失 -> 反向传播 -> 更新参数
        5. 每个epoch结束后打印训练损失和准确率
    """
    # 定义损失函数:交叉熵损失,适用于多分类问题
    criterion = nn.CrossEntropyLoss()

    # 定义优化器:Adam,学习率为lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 设置模型为训练模式,启用dropout和batch normalization的训练行为
    model.train()

    # 外层循环:遍历每个epoch
    for epoch in range(1, epochs + 1):
        # 初始化统计变量
        total = 0       # 统计总样本数
        correct = 0     # 统计正确预测数
        epoch_loss = 0.0  # 累计epoch总损失

        # 内层循环:遍历训练数据加载器中的每个batch
        for batch_x, batch_y in train_loader:
            # 将数据移动到指定的设备(CPU或GPU)
            batch_x = batch_x.to(device)  # 输入序列
            batch_y = batch_y.to(device)  # 标签(位置0-4)

            # 前向传播:将输入传入模型,得到logits
            logits = model(batch_x)

            # 计算损失:交叉熵损失衡量预测与真实标签的差异
            loss = criterion(logits, batch_y)

            # 清空梯度(因为PyTorch会累积梯度)
            optimizer.zero_grad()

            # 反向传播:计算梯度
            loss.backward()

            # 更新参数:根据梯度调整模型参数
            optimizer.step()

            # 累计损失(考虑batch大小)
            epoch_loss += loss.item() * batch_x.size(0)

            # 计算预测结果:取logits中最大值的索引作为预测类别
            preds = logits.argmax(dim=1)

            # 统计正确预测的数量
            correct += (preds == batch_y).sum().item()

            # 累计总样本数
            total += batch_x.size(0)

        # 计算本epoch的准确率
        acc = correct / total

        # 打印本epoch的训练统计信息
        # 格式: Epoch XX: loss=XXXX.XXXX acc=X.XXXX
        print(f"Epoch {epoch:02d}: loss={epoch_loss/total:.4f} acc={acc:.4f}")


# ============================================================
# 评估函数
# ============================================================

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    在测试集上评估模型性能

    参数:
        model: 要评估的神经网络模型
        test_loader: 测试数据加载器
        device: 计算设备(CPU或GPU)

    返回:
        float: 测试集上的准确率(0-1之间)
    """
    # 设置模型为评估模式,禁用dropout和batch normalization的训练行为
    model.eval()

    # 初始化统计变量
    correct = 0     # 统计正确预测数
    total = 0       # 统计总样本数

    # 使用torch.no_grad()上下文管理器,在评估时禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        # 遍历测试数据加载器中的每个batch
        for x, y in test_loader:
            # 将数据移动到指定设备
            x = x.to(device)  # 输入序列
            y = y.to(device)  # 标签

            # 前向传播:得到预测logits
            logits = model(x)

            # 取最大logit的索引作为预测类别
            preds = logits.argmax(dim=1)

            # 统计正确预测数
            correct += (preds == y).sum().item()

            # 累计总样本数
            total += x.size(0)

    # 返回准确率
    return correct / total


# ============================================================
# 模型保存函数
# ============================================================

def save_artifacts(model: nn.Module, vocab: List[str], char2idx: dict, path: str, extra: dict = None):
    """
    保存模型及相关数据到文件

    参数:
        model: 要保存的神经网络模型
        vocab: 词表列表,包含所有字符
        char2idx: 字符到索引的映射字典
        path: 保存文件的路径
        extra: 额外的元数据字典(可选),如模型类型、训练参数等
    """
    # 创建一个字典,包含模型状态、词表、字符映射和额外信息
    checkpoint = {
        "model_state": model.state_dict(),  # 模型的可学习参数
        "vocab": vocab,                      # 词表
        "char2idx": char2idx,                # 字符到索引的映射
        "extra": extra                       # 额外信息
    }

    # 使用torch.save将字典保存到指定路径
    torch.save(checkpoint, path)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数:程序入口

    流程:
        1. 解析命令行参数
        2. 创建训练和测试数据集
        3. 创建模型
        4. 训练模型
        5. 在测试集上评估模型
        6. 保存模型和相关信息

    命令行参数:
        --model: 模型类型,'rnn'或'lstm',默认'lstm'
        --train_size: 训练样本数量,默认10000
        --test_size: 测试样本数量,默认2000
        --epochs: 训练轮数,默认10
        --embed_dim: 词嵌入维度,默认32
        --hidden_dim: 隐藏层维度,默认64
        --batch: 批次大小,默认128
        --seed: 随机种子,默认42
        --save: 模型保存路径,默认'charpos_model.pth'
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加模型类型参数
    parser.add_argument(
        "--model",
        choices=["rnn", "lstm"],
        default="lstm",
        help="模型类型: 选择'rnn'或'lstm',默认'lstm'"
    )

    # 添加训练样本数量参数
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="训练样本数量,默认10000"
    )

    # 添加测试样本数量参数
    parser.add_argument(
        "--test_size",
        type=int,
        default=2000,
        help="测试样本数量,默认2000"
    )

    # 添加训练轮数参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数,默认10"
    )

    # 添加词嵌入维度参数
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="词嵌入维度,默认32"
    )

    # 添加隐藏层维度参数
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="隐藏层维度,默认64"
    )

    # 添加批次大小参数
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="批次大小,默认128"
    )

    # 添加随机种子参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子,默认42"
    )

    # 添加模型保存路径参数
    parser.add_argument(
        "--save",
        type=str,
        default="charpos_model.pth",
        help="模型保存路径,默认'charpos_model.pth'"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # ============================================================
    # 准备数据集
    # ============================================================

    # 定义字母表(26个小写英文字母)
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    # 创建训练数据集
    # 使用args.seed作为随机种子确保可复现
    train_ds = CharPosDataset(
        n_samples=args.train_size,
        seed=args.seed,
        alphabet=alphabet
    )

    # 创建测试数据集
    # 使用args.seed+1作为随机种子,确保测试集与训练集不同
    test_ds = CharPosDataset(
        n_samples=args.test_size,
        seed=args.seed + 1,
        alphabet=alphabet
    )

    # 创建训练数据加载器
    # batch_size: 每个batch的样本数
    # shuffle=True: 每次epoch打乱数据
    # drop_last=True: 如果最后一个batch不足batch_size,则丢弃
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True
    )

    # 创建测试数据加载器
    # shuffle=False: 不打乱数据,保证评估结果稳定
    # drop_last=False: 保留最后一个不足batch_size的batch
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False
    )

    # ============================================================
    # 创建模型
    # ============================================================

    # 确定计算设备:优先使用GPU,如果不可用则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例
    # vocab_size: 词表大小 = 26个字母 + 1个"你" = 27
    model = CharPosModel(
        vocab_size=len(train_ds.vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        model_type=args.model
    )

    # 将模型移动到指定设备上
    model.to(device)

    # 打印训练开始信息
    print("Starting training with model=", args.model)

    # ============================================================
    # 训练模型
    # ============================================================

    # 调用训练函数,使用学习率0.001
    train(model, train_loader, device, epochs=args.epochs, lr=0.001)

    # ============================================================
    # 评估模型
    # ============================================================

    # 在测试集上评估模型,获取测试准确率
    acc = evaluate(model, test_loader, device)

    # 打印测试准确率
    print(f"Test accuracy: {acc:.4f}")

    # ============================================================
    # 保存模型
    # ============================================================

    # 保存模型、词表和相关信息到指定路径
    save_artifacts(
        model,
        train_ds.vocab,
        train_ds.char2idx,
        args.save,
        extra={"model_type": args.model}
    )

    # 打印保存成功信息
    print(f"Saved artifacts to {args.save}")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 当脚本作为主程序运行时,调用main函数
    main()