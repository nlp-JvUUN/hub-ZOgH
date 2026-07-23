# coding:utf8
"""
第五周作业：训练基于Transformer的单向语言模型，完成文本生成
架构：Decoder-only Transformer（GPT风格）
任务：在中文语料上训练 next-token prediction，实现自回归文本生成

核心知识点对应PPT《如何得到LLM》：
  1. Causal Mask（下三角矩阵）→ 控制"谁能看见谁"
  2. 自回归生成 → 一个token一个token地续写
  3. 采样策略 → Greedy / Temperature / Top-K / Top-P
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# ===================== 1. 位置编码 =====================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    公式：PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ===================== 2. 带Causal Mask的多头自注意力 =====================
class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention（因果自注意力）
    核心：使用下三角Mask，每个token只能看到自己和之前的token
    这就是PPT中讲的"Decoder-only"架构的关键机制

    Mask矩阵示意（L=5）：
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ====== 关键：构造Causal Mask（下三角矩阵）======
        # mask[i][j] = 1 表示第i个token可以看见第j个token
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 将不能看到的位置设为 -inf，softmax后权重为0
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)

        # 拼接多头 + 输出变换
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        return output


# ===================== 3. 前馈网络 =====================
class FeedForward(nn.Module):
    """两层全连接 + GELU激活"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ===================== 4. Decoder Block =====================
class DecoderBlock(nn.Module):
    """
    一个Decoder Block = Causal Self-Attention + FeedForward
    结构：
    输入 -> [Causal Attention + 残差 + LayerNorm] -> [FFN + 残差 + LayerNorm] -> 输出
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Causal Self-Attention + 残差连接
        x = self.norm1(x + self.dropout1(self.attn(x)))
        # FeedForward + 残差连接
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


# ===================== 5. GPT语言模型 =====================
class MiniGPT(nn.Module):
    """
    迷你GPT语言模型（Decoder-only Transformer）

    结构：
    Token Embedding + Positional Encoding
        -> N x DecoderBlock
        -> Linear -> 输出概率分布（词表大小）

    训练目标：Next-Token Prediction
    给定前t个token，预测第t+1个token
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=256,
                 num_layers=3, max_len=256, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token Embedding：将token ID映射到d_model维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠N个Decoder Block
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终LayerNorm + 线性输出层
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len) — token ID序列
        返回: (batch_size, seq_len, vocab_size) — 每个位置的logits
        """
        batch_size, seq_len = input_ids.size()

        # Token Embedding + 位置编码
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过N个Decoder Block（内部带Causal Mask）
        for layer in self.layers:
            x = layer(x)

        # 最终归一化 + 线性映射到词表
        x = self.final_norm(x)
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)
        return logits


# ===================== 6. 文本生成（采样策略） =====================
def generate(model, start_tokens, max_new_tokens=100, temperature=1.0,
             top_k=0, top_p=0.9, device='cpu'):
    """
    自回归文本生成

    对应PPT《如何得到LLM》第四部分"采样策略"：
    1. Greedy: 永远选概率最大的 → temperature=0.01
    2. Temperature: T越小越确定，T越大越随机
    3. Top-K: 只从概率最高的K个中采样
    4. Top-P: 累积概率达到P的最小集合中采样
    """
    model.eval()
    tokens = start_tokens.clone().unsqueeze(0).to(device)  # (1, seq_len)

    for _ in range(max_new_tokens):
        # 截断到最大长度
        tokens_cond = tokens[:, -256:]

        # 前向传播，获取logits
        with torch.no_grad():
            logits = model(tokens_cond)

        # 取最后一个位置的logits
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Temperature缩放
        logits = logits / max(temperature, 1e-8)

        # Top-K过滤
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_values[:, -1].unsqueeze(1)
            logits[logits < threshold] = float('-inf')

        # Top-P (Nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过top_p的token
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # 还原顺序
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 拼接到序列末尾
        tokens = torch.cat([tokens, next_token], dim=1)

        # 如果生成了结束符，停止
        if next_token.item() == 2:  # [EOS] = 2
            break

    return tokens.squeeze(0).tolist()


def generate_greedy(model, start_tokens, max_new_tokens=100, device='cpu'):
    """贪心解码：每步选概率最大的token"""
    return generate(model, start_tokens, max_new_tokens,
                    temperature=0.01, top_k=1, top_p=1.0, device=device)


# ===================== 7. 数据准备 =====================
# 中文训练语料（覆盖多种句式，帮助模型学习语言模式）
CORPUS = """
人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
人工智能的研究包括机器人学、语言识别、图像识别、自然语言处理和专家系统等。
机器学习是人工智能的核心，它使计算机具有学习能力，而不需要进行明确的编程。
深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
神经网络是一种受人脑启发的计算模型，由相互连接的节点组成。
自然语言处理使计算机能够理解、解释和生成人类语言。
计算机视觉使计算机能够从图像和视频中获取高层次的理解。
语音识别技术将人类的语音转换为文本，广泛应用于智能助手。
自动驾驶汽车使用人工智能技术来感知环境并做出驾驶决策。
推荐系统通过分析用户行为来预测用户可能喜欢的内容。
数据挖掘是从大量数据中发现模式和知识的过程。
强化学习是机器学习的一个分支，智能体通过与环境交互来学习最优策略。
迁移学习是将在一个任务上学到的知识应用到另一个相关任务上。
生成对抗网络由生成器和判别器组成，可以生成逼真的图像和文本。
Transformer架构是现代自然语言处理的基础，它使用自注意力机制来处理序列数据。
自注意力机制允许模型在处理每个位置时关注输入序列的所有位置。
大型语言模型通过在海量文本数据上预训练来获得强大的语言理解能力。
预训练语言模型可以在下游任务上进行微调，以获得更好的性能。
对话系统使人机交互更加自然和高效，广泛应用于客服和智能助手。
知识图谱将实体和关系组织成图结构，支持复杂的语义查询。
"""

# 多样化补充语料
CORPUS_EXTRA = """
今天的天气真不错，阳光明媚，微风轻拂，适合出去散步。
我想去图书馆学习，因为那里安静而且有很多好书。
周末我计划和朋友一起去爬山，锻炼身体放松心情。
这家餐厅的菜很好吃，尤其是他们的招牌菜红烧肉。
我最近在学习编程，感觉Python是一门很有趣的语言。
时间过得真快，转眼间已经到了五月份。
科技的发展让我们的生活变得越来越便利。
阅读是一种很好的学习方式，可以开阔视野增长知识。
音乐能够放松心情，我经常在工作的时候听音乐。
运动对身体健康很重要，每天应该坚持锻炼至少三十分钟。
学习需要耐心和毅力，只有坚持不懈才能取得进步。
大自然的美丽让人叹为观止，我们应该保护环境。
创新是推动社会进步的重要力量，鼓励创新思维很有必要。
团队合作能够发挥每个人的优势，取得更好的成果。
良好的沟通是成功的关键，要学会倾听和表达。
"""


def build_vocab_and_data(corpus, seq_len=64):
    """
    构建词表和训练数据

    流程：
    1. 按字切分中文文本，建立字符级词表
    2. 将文本转为token ID序列
    3. 按固定长度切分为训练样本
    """
    # 按字符切分，建立词表
    chars = list(set(corpus))
    chars.sort()

    # 特殊token
    # 0: [PAD]  1: [UNK]  2: [EOS]
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2}
    for i, ch in enumerate(chars):
        vocab[ch] = i + 3

    vocab_size = len(vocab)
    id_to_char = {v: k for k, v in vocab.items()}

    # 全文转为token ID
    token_ids = [vocab.get(ch, 1) for ch in corpus]

    # 按seq_len切分训练样本
    # 对于语言模型：input是前n-1个token，label是后n-1个token（右移一位）
    samples = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[i:i + seq_len]
        if len(chunk) == seq_len:
            samples.append(chunk)

    return vocab, id_to_char, samples, vocab_size


# ===================== 8. 训练函数 =====================
def train_epoch(model, data, batch_size, optimizer, device):
    """训练一个epoch"""
    model.train()
    random.shuffle(data)
    total_loss = 0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if len(batch) < 2:
            continue

        batch_tensor = torch.LongTensor(batch).to(device)

        # 语言模型的输入和标签
        # input:  [t1, t2, t3, ..., t_{n-1}]
        # label:  [t2, t3, t4, ..., t_n]
        # 即：给定前文，预测下一个token
        input_ids = batch_tensor[:, :-1]
        labels = batch_tensor[:, 1:]

        # 前向传播
        logits = model(input_ids)

        # 计算交叉熵损失
        # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
        # labels: (batch, seq_len) -> (batch*seq_len,)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=0  # 忽略[PAD]
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_perplexity(model, data, batch_size, device):
    """评估困惑度（Perplexity）—— 越低越好"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < 2:
                continue

            batch_tensor = torch.LongTensor(batch).to(device)
            input_ids = batch_tensor[:, :-1]
            labels = batch_tensor[:, 1:]

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # 防止溢出
    return avg_loss, perplexity


# ===================== 9. 主程序 =====================
def main():
    # ---- 超参数 ----
    SEQ_LEN = 64        # 序列长度
    BATCH_SIZE = 8      # 批大小
    EPOCHS = 80         # 训练轮数
    LR = 3e-4           # 学习率
    D_MODEL = 128       # 模型维度
    NUM_HEADS = 4       # 注意力头数
    D_FF = 256          # FFN维度
    NUM_LAYERS = 3      # Decoder层数
    DROPOUT = 0.1       # Dropout率

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("第五周作业：基于Transformer的单向语言模型（MiniGPT）")
    print("=" * 60)

    # ---- 构建词表和数据 ----
    full_corpus = CORPUS + CORPUS_EXTRA
    vocab, id_to_char, train_data, vocab_size = build_vocab_and_data(full_corpus, SEQ_LEN)

    print(f"词表大小: {vocab_size}")
    print(f"训练样本数: {len(train_data)}")
    print(f"序列长度: {SEQ_LEN}")
    print(f"设备: {device}")

    # ---- 创建模型 ----
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"\n模型结构:")
    print(model)
    print("=" * 60)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- 训练 ----
    loss_history = []
    ppl_history = []

    print("\n开始训练...")
    print("-" * 60)
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_data, BATCH_SIZE, optimizer, device)
        val_loss, perplexity = evaluate_perplexity(model, train_data, BATCH_SIZE, device)
        scheduler.step()

        loss_history.append(train_loss)
        ppl_history.append(perplexity)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"第{epoch + 1:3d}轮 | Loss: {train_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f}")

    print("-" * 60)
    print(f"训练完成！最终 Loss: {loss_history[-1]:.4f}, Perplexity: {ppl_history[-1]:.2f}")

    # ---- 绘制训练曲线 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, EPOCHS + 1), loss_history, 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, EPOCHS + 1), ppl_history, 'r-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity (lower is better)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('AI/第五周/gpt_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存: AI/第五周/gpt_training_curves.png")

    # ---- 文本生成演示 ----
    print("\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)

    # 测试种子文本
    prompts = [
        "人工智能",
        "机器学习",
        "今天的天气",
        "深度学习",
        "自然语言处理",
    ]

    print("\n【1】Greedy 解码（确定性输出）")
    print("-" * 40)
    for prompt in prompts:
        input_tokens = torch.LongTensor([vocab.get(ch, 1) for ch in prompt])
        output_ids = generate_greedy(model, input_tokens, max_new_tokens=50, device=device)
        generated = ''.join([id_to_char.get(i, '?') for i in output_ids if i not in [0, 2]])
        print(f"  [{prompt}] -> {generated}")

    print("\n【2】Temperature + Top-P 采样（T=0.8, P=0.9）")
    print("-" * 40)
    for prompt in prompts:
        input_tokens = torch.LongTensor([vocab.get(ch, 1) for ch in prompt])
        output_ids = generate(model, input_tokens, max_new_tokens=50,
                              temperature=0.8, top_p=0.9, device=device)
        generated = ''.join([id_to_char.get(i, '?') for i in output_ids if i not in [0, 2]])
        print(f"  [{prompt}] -> {generated}")

    print("\n【3】高温度采样（T=1.2, P=0.95）—— 更有创造性")
    print("-" * 40)
    for prompt in prompts:
        input_tokens = torch.LongTensor([vocab.get(ch, 1) for ch in prompt])
        output_ids = generate(model, input_tokens, max_new_tokens=50,
                              temperature=1.2, top_p=0.95, device=device)
        generated = ''.join([id_to_char.get(i, '?') for i in output_ids if i not in [0, 2]])
        print(f"  [{prompt}] -> {generated}")

    # ---- 保存模型 ----
    model_path = "AI/第五周/mini_gpt.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': vocab_size,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'd_ff': D_FF,
            'num_layers': NUM_LAYERS,
            'max_len': SEQ_LEN,
        }
    }, model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
