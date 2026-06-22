大语言模型
bert快速回顾
预训练任务
1. MLM  随机 mask 15% 的 token，让模型根据上下文预测原词
2. NSP  判断句 B 是否是句 A 的下一句（二分类）
擅长任务
1. 文本分类（情感、主题）
2. 命名实体识别 NER
3. 抽取式问答（从原文找答案 span）
bert为什么不适合生成
1. 预训练目标不匹配：MLM 训练的是"完形填空"。生成任务需要的是"续写"，给定前文，预测下一个词。
2. 注意力是双向的：BERT 里每个位置可以看到序列中所有位置（包括它后面的词）。生成任务生成时第 t 步不应该看到 t+1 以后的词
3. 输出长度固定：MLM 的预测位置数 = 输入中 [MASK] 的个数，由输入结构决定。真正的生成需要模型自己决定：• 生成多长 • 什么时候停下（EOS）
根本原因：BERT 的架构和训练目标是为"理解"优化的，不是"生成"
Attention Mask
mask在attention中的softmax之前，把不允许看的位置的score设为-∞
mask矩阵的形状决定了模型做的任务：
- 全 1  全可见→ 完全双向    BERT     理解任务
- 下三角 → 因果生成           GPT、Llama     自回归任务
- 前缀双向+后缀因果 → 前缀 / 局部注意力       T5、早期GLM   条件生成 翻译任务 原文+译文、生成摘要 seq2seq 序列到序列的任务
换一个mask矩阵，同一个Transformer架构就能从 "理解模型" 变成 "生成模型"
[图片]
SFT有监督微调
预训练模型的训练目标是Next Token Prediction，模型学到的是"语料的分布规律"：给定前文, 最可能接下来出现什么？  它只是一个"概率续写器", 不是一个"问答助手"。
[图片]
 训练目标
仍然是 next-token prediction, 没有新结构。关键是label mask。 user 部分：只作为输入, 不计算 loss。assistant 部分：计算 loss, 反向传播更新参数
我们只想让模型学会「如何回答」, 而不是让它背熟用户提问。
[图片]
[图片]
[图片]
采样策略
在模型推理阶段起作用
解决模型推理阶段输出什么，下一个token怎么选
两种极端采样方式：
- 永远选概率最大的
  - 结果：完全确定, 完全可复现
  - 问题：死板, 重复, 缺乏创造力
  - 适合：代码生成、数学推理
- 按概率完全随机采样
  - 结果：多样但不可控
  - 问题：长尾 token 累积概率大, 容易胡言乱语
  - 直接用几乎不可行
Greedy Decoding  贪心
规则：每一步选概率最大的 token
优点：完全确定, 可复现
缺点：死板, 重复, 缺乏创造力
适合场景：需要确定答案的任务（代码、数学、结构化生成）
Beam Search  束搜索
规则：同时保留 k 条最优候选路径（beam width = k）, 选整体分数最高的序列
优点：平均质量比 greedy 高
缺点：计算开销 × k, 容易产出「平均答案」, 缺乏惊喜
适合场景：机器翻译、摘要（答案有相对标准）
Temperature  温度
一般和top-k和top-p搭配使用
[图片]
Top-K 采样
规则：只从概率最高的 K 个 token 中采样, 其余 token 的概率置零后重新归一化
优点：切掉长尾, 避免胡言乱语
缺点：K 是固定数, 不适配分布形态：尖锐分布 K=50 太宽松, 平坦分布 K=50 又太紧
Top-P / Nucleus 采样
规则：按概率从大到小排序, 选累积概率第一次达到 P 的最小集合, 再重新归一化采样
优点：候选数随分布形态自动调整
[图片]
代码示例

# 从零实现（示意）
import torch
import torch.nn.functional as F

def sample(logits, T=0.7,
           top_k=50, top_p=0.9):
    # 1. temperature
    logits = logits / T
    # 2. top-k
    v, _ = logits.topk(top_k)
    logits[logits < v[-1]] = -inf
    # 3. softmax → top-p
    probs = F.softmax(logits, -1)
    probs = apply_top_p(probs,top_p)
    # 4. 按概率采样
    return torch.multinomial(
        probs, num_samples=1)
# 用 HuggingFace transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m)
inputs = tok(prompt, return_tensors='pt')
output = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_tokens=256,
)
[图片]
LLM 的跨时代突破
LLM 与 BERT 的差距
[图片]
[图片]
传统NLP任务三大瓶颈（BERT及bert之间）
任务绑定：一个模型只能做一件事。情感分析模型不能做翻译，NER 模型不能做问答。每一项新需求都意味着一个全新的训练周期。
数据依赖：没有标注数据 = 没有能力。标注数据昂贵、稀缺、且难以覆盖长尾场景。低资源语言和专业领域尤为明显。
泛化边界：模型在训练分布外能力急剧下降。输入格式轻微改变、措辞不同都可能让效果大幅退化，对抗样本更是灾难性的。
In-Context Learning（ICL）
模型无需更新任何权重，仅凭 Prompt 中提供的若干示例，就能学会完成全新任务。
[图片]
Scaling Law
语言模型的性能与参数量、训练数据量、计算量之间存在稳定的幂律关系——三者增加，性能可预测地提升。
[图片]
参数量：模型中可学习权重的总数量。GPT-3 有 1750 亿参数，GPT-4 估计超过 1 万亿。参数越多，模型容量越大，能记忆和推理的知识越丰富。
数据量：训练语料的 Token 总数。GPT-3 用约 3000 亿 tokens，Llama-3 用超过 15 万亿 tokens。数据的多样性和质量同样关键。
算力：训练消耗的总浮点运算量，单位 FLOPs。通常用 GPU/TPU 小时计算。算力决定了你能实际使用多大规模的 N 和 D。
Chain-of-Thought（CoT）
在 Prompt 中提供带有中间推理步骤的示例，引导模型展示完整的推理过程，而不是直接跳到最终答案。
[图片]
[图片]
API生态
