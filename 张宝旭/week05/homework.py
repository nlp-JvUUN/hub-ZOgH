"""  # 文件头部文档字符串，说明这个脚本的用途
Simple character-level Transformer language model.  # 说明：这是一个字符级 Transformer 语言模型

Usage:  # 说明：下面给出命令行使用方法
    python transformer_lm.py --epochs 20  # 训练 20 轮
    python transformer_lm.py --epochs 20 --generate "hello"  # 训练后再根据提示词生成文本
"""  # 文档字符串结束

import argparse  # 导入命令行参数解析模块
import glob  # 导入文件匹配模块，用于读取多个文本文件
import math  # 导入数学模块，用于开方和指数运算
import os  # 导入操作系统模块，用于设置环境变量
import random  # 导入随机模块，用于打乱训练数据
import time  # 导入时间模块，用于统计训练耗时

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许 OpenMP 重复加载，绕过某些 Windows 环境下的库冲突

import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入神经网络模块，并简写为 nn
from torch.utils.data import DataLoader, Dataset  # 导入数据集基类和数据加载器


def load_corpus(pattern="*.txt"):  # 定义函数：按通配符读取语料文件
    texts = []  # 用列表保存每个文本文件的内容
    for path in glob.glob(pattern):  # 遍历所有匹配 pattern 的文件路径
        with open(path, encoding="utf-8", errors="ignore") as f:  # 用 UTF-8 打开文件，忽略坏字符
            texts.append(f.read())  # 读取整个文件内容并加入列表
    return "".join(texts)  # 把所有文本拼接成一个大字符串并返回


def build_vocab(text):  # 定义函数：根据语料构建字符词表
    chars = sorted(set(text))  # 取出去重后的所有字符，并排序，保证词表稳定
    char2idx = {c: i for i, c in enumerate(chars)}  # 构建“字符 -> 编号”的映射
    idx2char = {i: c for c, i in char2idx.items()}  # 构建“编号 -> 字符”的反向映射
    return char2idx, idx2char  # 返回两个映射表


class CharDataset(Dataset):  # 定义字符级数据集类，继承 PyTorch 的 Dataset
    def __init__(self, text, char2idx, seq_len):  # 初始化函数，输入文本、词表和序列长度
        self.seq_len = seq_len  # 保存序列长度，后面切片时会用到
        ids = [char2idx[c] for c in text if c in char2idx]  # 把文本中的每个字符转换成对应的整数 id
        self.data = torch.tensor(ids, dtype=torch.long)  # 把 id 列表转成 long 型张量

    def __len__(self):  # 定义数据集长度函数
        return max(0, len(self.data) - self.seq_len)  # 可取样本数 = 总长度 - 序列长度，最小为 0

    def __getitem__(self, idx):  # 定义按下标取样本的方法
        x = self.data[idx: idx + self.seq_len]  # 取一段长度为 seq_len 的输入序列
        y = self.data[idx + 1: idx + self.seq_len + 1]  # 目标序列是输入整体右移一位后的结果
        return x, y  # 返回输入和监督目标


class CausalSelfAttention(nn.Module):  # 定义因果自注意力模块
    def __init__(self, d_model, n_heads, dropout, max_seq_len):  # 初始化注意力层
        super().__init__()  # 调用父类初始化
        if d_model % n_heads != 0:  # 检查隐藏维度是否能被头数整除
            raise ValueError("d_model must be divisible by n_heads")  # 不能整除就报错

        self.n_heads = n_heads  # 保存注意力头数
        self.head_dim = d_model // n_heads  # 计算每个头的维度
        self.qkv = nn.Linear(d_model, 3 * d_model)  # 一次线性映射同时生成 Q、K、V
        self.proj = nn.Linear(d_model, d_model)  # 多头拼接后再投影回原维度
        self.attn_drop = nn.Dropout(dropout)  # 注意力权重上的 dropout
        self.resid_drop = nn.Dropout(dropout)  # 输出残差上的 dropout

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))  # 创建下三角矩阵，表示只能看见当前位置及之前位置
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))  # 注册为 buffer，使其随模型迁移到设备上但不参与训练

    def forward(self, x):  # 定义前向传播
        batch_size, seq_len, d_model = x.size()  # 取出输入张量的 batch、大写 T、隐藏维度
        qkv = self.qkv(x)  # 通过线性层把输入映射成拼接在一起的 QKV
        q, k, v = qkv.chunk(3, dim=-1)  # 在最后一维切成三块，分别得到 Q、K、V

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # 调整 Q 形状为 (B, H, T, D)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # 调整 K 形状为 (B, H, T, D)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # 调整 V 形状为 (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算缩放点积注意力分数
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))  # 用下三角 mask 把未来位置屏蔽掉
        att = torch.softmax(att, dim=-1)  # 对最后一维做 softmax，得到注意力概率
        att = self.attn_drop(att)  # 对注意力概率做 dropout

        out = att @ v  # 用注意力权重对 V 做加权求和
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # 把多头结果拼回原来的隐藏维度
        return self.resid_drop(self.proj(out))  # 经过输出投影和 dropout 后返回


class TransformerBlock(nn.Module):  # 定义一个 Transformer Block
    def __init__(self, d_model, n_heads, ff_dim, dropout, max_seq_len):  # 初始化模块参数
        super().__init__()  # 调用父类初始化
        self.ln1 = nn.LayerNorm(d_model)  # 第一层 LayerNorm，放在注意力之前
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)  # 因果自注意力子层
        self.ln2 = nn.LayerNorm(d_model)  # 第二层 LayerNorm，放在前馈网络之前
        self.ffn = nn.Sequential(  # 定义前馈网络
            nn.Linear(d_model, ff_dim),  # 先从 d_model 扩到 ff_dim
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Linear(ff_dim, d_model),  # 再从 ff_dim 投影回 d_model
            nn.Dropout(dropout),  # 前馈网络输出后做 dropout
        )  # 前馈网络定义结束

    def forward(self, x):  # 定义一个 block 的前向传播
        x = x + self.attn(self.ln1(x))  # 先归一化，再做注意力，然后和输入做残差相加
        x = x + self.ffn(self.ln2(x))  # 再归一化，过前馈网络，然后继续做残差相加
        return x  # 返回当前 block 的输出


class TransformerLM(nn.Module):  # 定义完整的 Transformer 语言模型
    def __init__(self, vocab_size, d_model, n_heads, ff_dim, num_layers, max_seq_len, dropout):  # 初始化模型
        super().__init__()  # 调用父类初始化
        self.max_seq_len = max_seq_len  # 保存最大序列长度，生成和训练时都会检查
        self.token_embed = nn.Embedding(vocab_size, d_model)  # token embedding，把字符 id 映射成向量
        self.pos_embed = nn.Embedding(max_seq_len, d_model)  # 位置 embedding，给每个位置一个向量
        self.drop = nn.Dropout(dropout)  # 输入层 dropout
        self.blocks = nn.ModuleList(  # 用 ModuleList 保存多层 Transformer Block
            [  # 下面是构建每一层 block 的列表推导式
                TransformerBlock(d_model, n_heads, ff_dim, dropout, max_seq_len)  # 创建一个 block
                for _ in range(num_layers)  # 一共创建 num_layers 层
            ]  # 列表推导式结束
        )  # ModuleList 定义结束
        self.ln_f = nn.LayerNorm(d_model)  # 最后一层 LayerNorm
        self.head = nn.Linear(d_model, vocab_size)  # 输出层，把隐藏状态映射到词表大小的 logits

    def forward(self, x):  # 定义模型前向传播
        _, seq_len = x.size()  # 取出输入序列长度
        if seq_len > self.max_seq_len:  # 如果输入长度超过模型允许的最大长度
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")  # 直接报错提示

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # 构造位置 id，形状为 (1, T)
        x = self.token_embed(x) + self.pos_embed(positions)  # token embedding 和 position embedding 相加
        x = self.drop(x)  # 对输入表示做 dropout

        for block in self.blocks:  # 依次通过每一个 Transformer Block
            x = block(x)  # 更新隐藏状态

        x = self.ln_f(x)  # 通过最后一层 LayerNorm
        return self.head(x)  # 输出每个位置对下一个字符的预测 logits

    @torch.no_grad()  # 生成时不需要计算梯度
    def generate(self, idx, max_new_tokens, temperature=1.0):  # 定义自回归生成函数
        self.eval()  # 切换到评估模式，关闭 dropout
        for _ in range(max_new_tokens):  # 循环生成指定数量的新 token
            idx_cond = idx[:, -self.max_seq_len:]  # 如果上下文太长，只保留最近的 max_seq_len 个 token
            logits = self(idx_cond)  # 前向计算得到 logits
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # 只取最后一个位置的预测，并用 temperature 调整分布尖锐程度
            probs = torch.softmax(logits, dim=-1)  # 把 logits 转成概率
            next_token = torch.multinomial(probs, num_samples=1)  # 按概率采样得到下一个 token
            idx = torch.cat([idx, next_token], dim=1)  # 把新 token 接到原序列后面
        return idx  # 返回完整的生成序列


def run_epoch(model, loader, criterion, optimizer, device, train=True, log_interval=100):  # 定义单轮训练/验证函数
    model.train(train)  # 如果 train=True 则进入训练模式，否则进入评估模式
    total_loss = 0.0  # 累计总 loss，用于最后求平均
    total_tokens = 0  # 累计 token 数，用于计算平均 loss
    start_time = time.time()  # 记录这一轮开始的时间
    phase = "train" if train else "val"  # 记录当前阶段名称，便于打印日志

    for step, (x, y) in enumerate(loader, start=1):  # 从 DataLoader 里逐批取出输入和目标，并记录当前是第几个 batch
        x, y = x.to(device), y.to(device)  # 把输入和目标移动到 CPU 或 GPU
        logits = model(x)  # 前向计算得到预测 logits
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # 展平后计算交叉熵损失

        if train:  # 如果当前是训练阶段
            optimizer.zero_grad()  # 先清空上一轮梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 用优化器更新参数

        total_loss += loss.item() * y.numel()  # 按 token 数累计总损失
        total_tokens += y.numel()  # 累计总 token 数

        if step % log_interval == 0 or step == len(loader):  # 每隔固定 batch 数，或者在最后一个 batch 时打印进度
            avg_loss = total_loss / total_tokens  # 计算到当前为止的平均 loss
            ppl = math.exp(avg_loss)  # 计算到当前为止的困惑度
            elapsed = time.time() - start_time  # 计算已经过去的时间
            print(  # 打印当前阶段进度
                f"[{phase}] step {step}/{len(loader)}  "
                f"avg_loss={avg_loss:.4f}  ppl={ppl:.2f}  "
                f"elapsed={elapsed:.1f}s"
            )

    avg_loss = total_loss / total_tokens  # 计算平均每个 token 的 loss
    ppl = math.exp(avg_loss)  # 困惑度 PPL = exp(loss)
    return avg_loss, ppl  # 返回平均损失和困惑度


@torch.no_grad()  # 生成文本时不需要梯度
def generate_text(model, prompt, char2idx, idx2char, device, max_new_tokens, temperature):  # 定义辅助生成函数
    prompt_ids = [char2idx[c] for c in prompt if c in char2idx]  # 把提示词中的字符转成 id，未登录字符直接跳过
    if not prompt_ids:  # 如果提示词里没有任何字符出现在词表中
        prompt_ids = [0]  # 就退化成用词表第一个字符作为起点

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # 把提示词 id 变成形状为 (1, T) 的张量
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()  # 调用模型生成并转回 Python 列表
    return "".join(idx2char[i] for i in out)  # 把生成出的 id 序列重新解码成字符串


def main():  # 主函数，负责解析参数、训练模型、保存模型、可选生成文本
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--epochs", type=int, default=20)  # 训练轮数
    parser.add_argument("--seq_len", type=int, default=128)  # 每个训练样本的序列长度
    parser.add_argument("--batch_size", type=int, default=64)  # 批大小
    parser.add_argument("--d_model", type=int, default=256)  # Transformer 隐藏维度
    parser.add_argument("--n_heads", type=int, default=4)  # 多头注意力头数
    parser.add_argument("--ff_dim", type=int, default=1024)  # 前馈网络中间层维度
    parser.add_argument("--num_layers", type=int, default=4)  # Transformer block 层数
    parser.add_argument("--dropout", type=float, default=0.1)  # dropout 比例
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率
    parser.add_argument("--val_ratio", type=float, default=0.05)  # 验证集占比
    parser.add_argument("--corpus", default="*.txt")  # 语料文件匹配模式
    parser.add_argument("--save", default="transformer_best_model.pt")  # 最优模型保存路径
    parser.add_argument("--generate", default="")  # 可选：训练后输入提示词进行生成
    parser.add_argument("--max_new_tokens", type=int, default=100)  # 最多生成多少个新字符
    parser.add_argument("--temperature", type=float, default=1.0)  # 生成时采样温度
    parser.add_argument("--log_interval", type=int, default=50)  # 每隔多少个 batch 打印一次训练进度
    args = parser.parse_args()  # 解析命令行参数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU
    print(f"device: {device}  model: TRANSFORMER")  # 打印当前运行设备和模型类型

    text = load_corpus(args.corpus)  # 按给定模式读取所有语料文本
    if not text:  # 如果语料为空
        raise FileNotFoundError("No .txt files were found. Please check the corpus path.")  # 报错提示没有找到语料
    print(f"corpus chars: {len(text):,}")  # 打印语料总字符数

    char2idx, idx2char = build_vocab(text)  # 根据语料构建词表
    vocab_size = len(char2idx)  # 计算词表大小
    print(f"vocab size: {vocab_size}")  # 打印词表大小

    lines = text.splitlines()  # 按行切分语料，便于做训练/验证划分
    random.shuffle(lines)  # 随机打乱各行顺序
    split = int(len(lines) * (1 - args.val_ratio))  # 按比例计算训练集和验证集分割点
    train_text = "\n".join(lines[:split])  # 把前半部分行重新拼成训练文本
    val_text = "\n".join(lines[split:])  # 把后半部分行重新拼成验证文本

    train_ds = CharDataset(train_text, char2idx, args.seq_len)  # 创建训练数据集
    val_ds = CharDataset(val_text, char2idx, args.seq_len)  # 创建验证数据集

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)  # 创建训练数据加载器，按 batch 打乱
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)  # 创建验证数据加载器，不打乱

    model = TransformerLM(  # 创建 Transformer 语言模型实例
        vocab_size=vocab_size,  # 传入词表大小
        d_model=args.d_model,  # 传入隐藏维度
        n_heads=args.n_heads,  # 传入头数
        ff_dim=args.ff_dim,  # 传入前馈层维度
        num_layers=args.num_layers,  # 传入层数
        max_seq_len=args.seq_len,  # 传入最大序列长度
        dropout=args.dropout,  # 传入 dropout 比例
    ).to(device)  # 把模型移动到指定设备

    total_params = sum(p.numel() for p in model.parameters())  # 统计模型总参数量
    print(f"model params: {total_params:,}")  # 打印参数量

    criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 定义优化器为 AdamW

    best_val_ppl = float("inf")  # 记录当前最好的验证集 PPL，初始设为无穷大

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")  # 打印训练日志表头
    print("-" * 56)  # 打印分隔线

    for epoch in range(1, args.epochs + 1):  # 从第 1 轮训练到第 epochs 轮
        epoch_start = time.time()  # 记录当前 epoch 开始时间
        tr_loss, tr_ppl = run_epoch(  # 跑一轮训练，得到训练损失和训练 PPL
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            log_interval=args.log_interval,
        )
        with torch.no_grad():  # 验证时关闭梯度
            va_loss, va_ppl = run_epoch(  # 跑一轮验证，得到验证损失和验证 PPL
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                log_interval=args.log_interval,
            )

        marker = "  *" if va_ppl < best_val_ppl else ""  # 如果当前验证结果更好，就打一个星号标记
        if va_ppl < best_val_ppl:  # 如果当前验证 PPL 刷新了最好成绩
            best_val_ppl = va_ppl  # 更新最好验证 PPL
            torch.save(  # 保存 checkpoint
                {  # checkpoint 内容开始
                    "model_state": model.state_dict(),  # 保存模型参数
                    "char2idx": char2idx,  # 保存字符到编号的映射
                    "idx2char": idx2char,  # 保存编号到字符的映射
                    "args": vars(args),  # 保存训练时用到的参数
                },  # checkpoint 字典结束
                args.save,  # 保存到指定文件
            )  # 保存结束

        epoch_time = time.time() - epoch_start  # 计算当前 epoch 总耗时
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}  time={epoch_time:.1f}s")  # 打印当前轮次结果

    print(f"\ntraining finished. best val PPL: {best_val_ppl:.2f}  saved to {args.save}")  # 训练完成后打印最终结果

    if args.generate:  # 如果用户传入了生成提示词
        sample = generate_text(  # 调用生成辅助函数
            model,  # 传入训练后的模型
            prompt=args.generate,  # 传入提示词
            char2idx=char2idx,  # 传入词表映射
            idx2char=idx2char,  # 传入反向词表映射
            device=device,  # 传入设备
            max_new_tokens=args.max_new_tokens,  # 传入最大生成长度
            temperature=args.temperature,  # 传入采样温度
        )  # 生成函数调用结束
        print("\ngenerated text:")  # 打印提示文字
        print(sample)  # 打印生成结果


if __name__ == "__main__":  # 如果这个文件是作为主程序直接运行
    main()  # 调用主函数
