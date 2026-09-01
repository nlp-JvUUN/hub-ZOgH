# Week16 作业：没讲过的开源模型结构调研

> 作业主题：**调研几个没讲的开源模型的结构特点**。
> 调研口径：截至 2026-08-20；优先采用官方技术报告、官方 GitHub README、官方模型卡与 Hugging Face 模型卡。
> 本作业的交付物是一份可直接拆成汇报页的调研报告：**6 个模型 × 3 条结构支线 × 1 张横向对比表**。

---


| 模型 | 发布机构 | 结构路线（一句话） | 与课件已讲内容的差异 |
|------|---------|-------------------|---------------------|
| Mamba-2 | 普林斯顿 + AI2 等 | 状态空间模型（SSM） | 课件讲的是"注意力家族的压缩"，Mamba-2 是**另一条家族**：用结构化状态空间递推代替注意力 |
| RWKV-6 | RWKV 社区 | RNN 形态的线性注意力 | 课件讲了"线性注意力混合"，但 RWKV 把整个 Transformer 变成**单层递推 RNN**，训练推理同构 |
| Jamba | AI21 Labs | Mamba + Transformer + MoE 三层混合 | 课件分别讲了线性注意力与 MoE，Jamba 是**首个把两者塞进同一个生产模型的工程验证** |
| Grok-1 | xAI | 超大稀疏 MoE（314B） | 课件以细粒度 MoE 为主，Grok-1 是**粗粒度 8 专家 Top-2** 的对照样本，且权重全量开放 |
| OLMoE | AI2 | 全开放研究型 MoE（约 1.3B 激活） | 把细粒度 MoE（64 专家 Top-8）做成**6.9B 总参/约 1.3B 激活的最小可复现配方**，权重/数据/代码/日志全开放，路由技巧均有消融实验 |
| Gemma 2 | Google | Dense 路线的结构级效率 | 课件默认 dense 是基线，Gemma 2 证明**不改 MoE 也能靠结构细节（logit cap、交替注意力）提效** |


## 调研主线：三条结构支线 × 六个模型

本周课件的五条演进轴线（注意力复杂度、FFN 容量、残差稳定化、位置记忆、多模态耦合）之外，
本作业新增一条更根本的轴线：**"注意力"本身是否可以被替换？**

```
支线 A：线性时间架构（不计算 QK^T 也能建模长程依赖）
   Mamba-2 —— 结构化状态空间（SSD）：把 SSM 与注意力统一到同一数学框架
   RWKV-6  —— RNN 式线性注意力：WKV 递推 + token shift，训练=推理同构
   Jamba    —— 混合架构：1:7 的 Mamba:Transformer 配比 + MoE，生产级验证

支线 B：MoE 的两种开源配方（"稀疏激活"的两个极端）
   Grok-1 —— 314B 总参数 / 8 专家 Top-2：粗粒度、全量开源
   OLMoE  —— 6.9B 总参数 / 约 1.3B 激活 / 64 专家 Top-8：细粒度、全开放

支线 C：Dense 路线的结构级效率（不改稀疏性，改结构细节）
   Gemma 2 —— logit cap 稳定训练 + 交替局部/全局注意力省算力
```

后续每个模型按固定模板展开：**模型概况 → 核心结构图解 → 结构机制 → 与课件已讲技术对比 → 亮点与局限 → 一句话结论**。

---

## 三、支线 A：线性时间架构（不计算 QKᵀ 也能建模长程依赖）

### 1. Mamba-2：用"结构化状态空间（SSD）"把 SSM 与注意力统一

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | 2024-05（论文 arXiv 提交）/ 2024-06（官方模型随仓库发布），Albert Gu / Tri Dao 领衔（普林斯顿、AI2、Cartesia 等） |
| 论文 | *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*（arXiv:2405.21060） |
| 模型系列 | Mamba-2-130M / 370M / 780M / 1.3B / 2.7B（官方发布范围；另有 Mamba-2-Hybrid 混合注意力变体） |
| 结构 | 纯 SSM 网络：**没有一层 softmax attention**；每个 SSD 层 = 选择性状态空间递推 + 门控 MLP（expansion=2）+ 残差/归一化 |
| 关键配置 | 多头 SSM 结构：QK 头维度 = d_state=16，V 头维度 = d_head=64；A 矩阵参数化为每头共享的标量时间常数，计算可写成大矩阵乘法（matmul 友好） |
| 上下文 | 状态为常数大小、理论无限；官方模型按 4K 训练，推理时状态不随序列长度增长 |
| 许可 | Apache 2.0 |

**核心结构图解**

```
Mamba-2 块（替代标准注意力的位置）：

x ─→ RMSNorm ─→ 输入投影（多输入SSM，类GQA共享投影）
                    │
                    ▼
          分块算法（SSD 对偶视角）：
    ┌───────────────────────────────┐
    │ 块内：结构化掩码注意力（类似线性注意力的块内计算，Tensor Core 友好）│
    │ 块间：常数大小的状态 (A,B,C) 递推                               │
    └───────────────────────────────┘
                    │
                    ▼
      输出投影 ─→ ⊕（残差）
```

**结构机制**

1. **SSD 对偶性（核心理论贡献）**：论文证明"标量对角选择性 SSM"（Mamba-1 采用的那类参数化）与"结构化掩码注意力（SMA）"在数学上等价——SSM 的转移矩阵可以写成带结构化掩码的低秩形式。结论：*注意力是 SSM，SSM 也是注意力*，两类架构统一在 SSD 框架下。
2. **分块分解算法**：训练/预填充时把序列按块（块大小 16/64/128）切分——**块内**用类注意力的并行矩阵乘法一次算完（复用高效 GEMM），**块间**做一次状态传递（SSM 式顺序递推）。训练复杂度从 O(L²) 降为准线性 O(L·d)，官方提供融合 CUDA/Triton 内核。相比 Mamba-1 训练速度快约 2–8×。
3. **多头结构带来的硬件效率**：把 Mamba-1 的标量对角 SSM 参数化为"每头共享标量时间常数 + 多头分组"（QK 头维度 16、V 头维度 64），与注意力头结构一一对应，整个计算变成大矩阵乘法——这是训练吞吐接近 FlashAttention-2 版 Transformer 的关键。

**与课件已讲技术的对比**

课件讲的"线性注意力混合"（DSA、Gated DeltaNet 等）是在 **Transformer 内部**把部分层换成线性注意力；Mamba-2 是**完全不使用 attention 的另一条家族**，并给出了"SSM = 结构化注意力"的统一数学框架。MLA 压缩的是 KV cache（仍是注意力计算），Mamba-2 压缩的是"历史本身"（常数状态递推）。

**亮点与局限**

- 亮点：O(T) 时间/空间复杂度；理论统一（SSD）；块分解让训练能吃满 Tensor Core（2.7B 吞吐可与 FlashAttention-2 版 Transformer 相当）；2.7B 模型在常识/推理基准上与同规模 Transformer 相当（论文对比口径）。
- 局限：论文明确指出纯 SSM 的**精确检索与 in-context learning 弱于 softmax attention**（状态是有损压缩）；需要专门 kernel，生态依赖 mamba-ssm 等库。

**一句话结论**：Mamba-2 用结构化状态空间把注意力从二次方复杂度里解放出来，并证明了"SSM 与注意力等价"——但代价是检索精度，这为后面的混合架构埋下伏笔。

### 2. RWKV-6（Finch）：把 Transformer 折叠成一条 RNN 递推

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | RWKV 社区（彭博主导），论文 2024-04（arXiv:2404.05892），RWKV-6 "Finch" 系列随论文发布、14B 于 2024-10 发布 |
| 模型系列 | RWKV-6 "World" 系列：1.6B（24 层 / 2560 维）、3B（32 层 / 3072 维）、7B（32 层 / 4096 维）、14B（40 层 / 5120 维） |
| 结构 | RNN 形态：**没有注意力**，由 time mixing（WKV 递推）+ channel mixing（门控 FFN）两类子层交替堆叠 |
| 关键配置 | **矩阵值状态**（d×d）；数据依赖的逐通道衰减 w = exp(−exp(decay(x)))（由输入动态生成）；数据依赖的 token shift（tiny attention 动态计算混合系数）；训练按 chunk 并行、推理单步递推，**训练与推理架构同构** |
| 上下文 | 基座 4K、官方微调可到 128K+；状态常数大小、理论无限；纯 CPU 也能跑 14B 推理 |
| 许可 | MIT（RWKV-LM 仓库） |

**核心结构图解**

```
RWKV-6 块（每个 token 一步递推）：

xₜ ─→ LayerNorm ─→ token shift ─→ time mixing：WKV 递推 ─→ ⊕
   └→ LayerNorm ─→ token shift ─→ channel mixing：门控 FFN ─→ ⊕

WKV 递推（核心公式，arXiv:2404.05892）：
wkvₜ = diag(u)·kₜᵀ·vₜ + Σᵢ₌₁ᵗ⁻¹ diag(⊙ⱼ₌ᵢ₊₁ᵗ⁻¹ wⱼ)·kᵢᵀ·vᵢ
    ↑ 当前 token       ↑ 历史 token，每个通道乘上"学到的衰减因子 w"
```

**结构机制**

1. **WKV = 带遗忘的线性注意力**：把注意力中的 softmax 换成**逐通道可学习衰减**——w = exp(−exp(decay(x))) 由输入动态生成（数据依赖），历史信息按通道以不同速率衰减；u 是当前 token 的可学习加权（bonus）。数学上与门控线性注意力一族（Gated DeltaNet 等）等价，是"带可学习遗忘门的 RNN"。
2. **数据依赖的 token shift**：把上一层的"当前 token 与前一 token"表示做线性混合再送入子层（近似一阶差分），RWKV-6 中这个混合系数由一个小型注意力模块（tiny attention）按输入动态计算——**连时序注入都是数据依赖的**。
3. **训练/推理同构**：训练时可按 chunk 并行 + WKV 并行扫描（线性复杂度）；推理时退化为单步状态递推——没有 KV cache，状态就是模型本身（对比 Transformer：KV cache 只是"缓存"，RWKV 的状态是"记忆本体"）。

**与课件已讲技术的对比**

课件讲了"线性注意力混合"与 MLA 等**注意力家族**的压缩；RWKV-6 是**RNN 家族**的代表：把整棵 Transformer 树折叠成一条链，用"数据依赖衰减"替代 softmax 归一化。它与 Mamba 同属线性时间大族，但实现形态截然不同（Mamba 用连续状态空间 + 卷积/扫描 kernel；RWKV 用离散 WKV 递推 + token shift），且训练/推理同构这一点比 Mamba 更彻底。

**亮点与局限**

- 亮点：推理开销极低（无 KV cache、可 CPU 推理）、训练/推理同构、超长上下文、权重/代码/训练数据全开放。
- 局限：检索/ICL 精度仍是短板；**矩阵值状态每层占用 O(d²) 内存**（7B 约 4096² 浮点/层），长序列推理显存可观；数据依赖衰减使跨 token 并行需要专门 CUDA WKV kernel；同规模质量与 Transformer 旗舰仍有差距（14B 参数才摸到 7B 级 Transformer 的能力区间，社区评测口径）。

**一句话结论**：RWKV-6 用一条"带可学习遗忘的 RNN 递推"换来了极低的推理成本与训练/推理同构——是"把注意力从架构里拿掉"的最激进样本之一。

### 3. Jamba：Mamba + Transformer + MoE 的三层混合（生产级验证）

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | AI21 Labs，2024-03（论文 *Jamba: A Hybrid Transformer-Mamba Language Model*，arXiv:2403.19887） |
| 规模 | **52B 总参数 / 12B 激活**；原生 256K 上下文 |
| 层配置 | 共 **64 层**：1:7 配比 → 8 层注意力 + 56 层 Mamba；隐藏维度 4096（官方 config） |
| 注意力层细节 | GQA（32 头 / 8 KV 头）+ **滑动窗口注意力**（窗口 4096）+ RoPE |
| MoE | FFN 采用 **8 个专家、Top-2 路由**（每次激活 2 个专家） |
| 后续家族 | Jamba-1.5（2024-08，论文 arXiv:2408.12570）：Mini = 94B 总 / 12B 激活，Large = 398B 总 / 32B 激活；均升级为 **32 专家 + 2 共享专家、Top-1 路由**，256K 上下文 |
| 许可 | Jamba v0.1 为 Apache 2.0；Jamba-1.5 为 Jamba Open Model License（需同意条款） |

**核心结构图解**

```
Jamba 层堆叠（每 8 层一组，1:7 配比）：

[Mamba][Mamba][Mamba][Mamba][Mamba][Mamba][Mamba][Attention+MoE]
 └────── 7 层线性递推：负责长程建模、线性复杂度 ──────┘ └ 1 层精确注意力：负责检索锚点
                                                           └ 8 专家 Top-2：稀疏扩容参数

256K 上下文 = Mamba 的线性长程能力 + 少数 attention 层兜底精确检索 + MoE 撑起参数量
```

**结构机制**

1. **为什么混合**：纯 Mamba 检索/ICL 弱（见模型 1 的局限），纯 Transformer 注意力二次方。Jamba 用 **1:7 配比**：大部分层线性（省算力、撑起 256K 上下文），少部分 attention 层当"精确检索锚点"（且用滑动窗口注意力进一步降计算），MoE 负责在不增加激活成本的前提下扩容参数。
2. **MoE 叠加**：注意力层的 FFN 换成 8 专家 Top-2，12B 激活撑起 52B 总参数——混合架构与稀疏激活互相叠加；官方报告其长上下文吞吐约为 Mixtral-8x7B 的 3 倍。
3. **生产级验证**：Jamba 是**首个投入生产使用的 Mamba 混合模型**（2024 年 3 月发布即上线 API），随后家族化推出 Jamba-1.5（Mini/Large：专家数增至 32+2 共享、路由改为 Top-1+共享专家以降低激活成本），证明该配方可规模化。

**与课件已讲技术的对比**

课件分别讲了线性注意力（DSA/CSA/HCA）与细粒度 MoE，但都是"单点技术"；Jamba 回答的是**工程问题**：SSM、注意力、MoE 三者如何按什么比例组装成一个能上线的模型。1:7 不是理论常数，而是"速度、检索质量、硬件利用率"之间的经验折中——这与 MiniMax-01 的 7:1 线性:softmax。

**亮点与局限**

- 亮点：原生 256K 上下文；长序列吞吐高（线性层占 7/8）；混合配方可复现、可家族化。
- 局限：需要两套 kernel（Mamba 扫描 + flash attention），工程复杂度高；线性层的检索短板仍存在，配比是经验调出来的；52B 权重需申请获取，非完全开放。

**一句话结论**：Jamba 用"7 份线性 + 1 份精确 + MoE 扩容"把三条支线焊进一个生产模型——混合架构从此从论文走向产品。

---

## 四、支线 B：MoE 的两种开源配方（"稀疏激活"的两个极端）

### 4. Grok-1：314B 粗粒度 MoE 的全量开源

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | xAI，2024-03-17 以 Apache-2.0 开源权重（当时最大规模开源模型之一） |
| 规模 | **314B 总参数**；8 专家、Top-2 路由（粗粒度，无共享专家）；64 层；隐藏维度 6,144 |
| 注意力 | GQA：48 query 头 / 8 KV 头；RoPE（base 1e4）；RMSNorm；GeGLU 激活 |
| 词表/上下文 | 词表 131,072（SentencePiece）；上下文 8,192 |
| 激活参数 | 官方未明确公布；按 8 专家 Top-2 估算，每 token 激活约 25% 的专家参数（社区口径） |
| 开放度 | 权重 + 推理代码（JAX 实现）全开放；**无论文、无训练代码/训练数据**；开源的是未做指令微调的基座模型 |

**核心结构图解**

```
Grok-1 每一层（共 64 层）：

[ 注意力：GQA（48 query / 8 KV 头）+ RoPE ] → [ MoE-FFN：8 专家（无共享）、Top-2 路由 ]
                                                   └ 每个 token 只激活 2/8 的专家

总参数 314B，官方 README 称约 25% 的权重是注意力权重；
每个 token 的 FFN 计算只经过 2/8 专家 → 稀疏激活：总参数很大，单 token 计算量远小于 dense 314B
```

**结构机制**

1. **粗粒度 MoE 对照样本**：8 个"大专家"、Top-2、无共享专家——与 DeepSeek 式"160+ 细粒度小专家"形成两极。专家越少越大，路由与通信越简单，但组合粒度粗。
2. **稀疏激活的收益**：314B 总参数提供容量，Top-2 控制单 token 计算量（专家 FFN 只激活 25%），推理成本远小于同参数量的 dense 模型。
3. **开源的意义**：当时**最大的全量开放权重**之一，研究者和社区第一次能直接拿到 300B 级 MoE 的完整结构做实验（如蒸馏、路由分析、量化）；但也正因为只开源了推理代码，训练细节（是否用辅助负载均衡损失等）官方未披露，无法确认。

**与课件已讲技术的对比**

课件以**细粒度** MoE 为主线（DeepSeek 的 256/896 专家、无辅助损失路由）；Grok-1 是**粗粒度**对照：8 专家 Top-2、64 层、全量开源。它回答的问题是"不追求细粒度，粗粒度 MoE 能不能撑起 300B 级容量"——为 MoE 粒度这个连续谱提供了一个真实的大模型端点。

**亮点与局限**

- 亮点：314B 全量开放；粗粒度 MoE 工程简洁（8 专家通信开销低）；是研究"专家粒度 × 路由质量"的天然样本。
- 局限：上下文仅 8K（远低于同期旗舰）；无论文、无训练数据/细节，无法复现；BF16 推理需约 8 张 H100-80GB 级显存，部署门槛高；开源的是基座模型（无指令微调），且官方自述推理代码"以正确性而非效率为目标"。

**一句话结论**：Grok-1 证明"8 个专家也能撑起 300B 级模型"，并把 314B 权重的完整结构交给了社区——粗粒度 MoE 的活标本。

### 5. OLMoE：把细粒度 MoE 做成约 1.3B 激活的全开放研究配方

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | AI2（艾伦人工智能研究所），2024-09/10（论文 *OLMoE: Open Mixture-of-Experts Language Models*，arXiv:2409.02060） |
| 规模 | **OLMoE-1B-7B：6.9B 总参数 / 约 1.3B 激活**；16 层；隐藏维度 2,048；64 个专家（无共享专家）、Top-8 路由 |
| 路由细节 | dropless **token-choice** 路由（每个 token 固定选 8 个专家、不丢 token）；**预训练使用负载均衡辅助损失（权重 0.01）+ router z-loss（权重 0.001）**，仅 SFT 阶段去掉负载均衡损失 |
| 其他结构 | SwiGLU；RMSNorm + **QK-Norm**；RoPE（base 1e4）；输入/输出 embedding **不共享**（tie_word_embeddings: false）；词表 50,304；上下文 4,096 |
| 训练 | 5T tokens（Dolma 1.7 等组成的 OLMoE-Mix）；**从零训练**（非 dense→MoE 升级）；同时发布同数据同规模的 dense 对照模型做严格受控对比 |
| 开放度 | **权重、训练数据、代码、日志全开放**（Apache 2.0） |

**核心结构图解**

```
OLMoE-1B-7B 单层：

[ 注意力（MHA，16 头）+ QK-Norm ] → [ MoE-FFN：64 个细粒度专家，Top-8 路由 ]
                                        └ 约 1.3B 激活 / 6.9B 总参数：8/64 的专家被激活

与旗舰细粒度 MoE 相同的配方，缩到"约 1.3B 激活"尺寸 → 单卡可跑、快速迭代
```

**结构机制**

1. **旗舰配方的下放**：细粒度专家 + Top-8 路由（C(64,8)≈44 亿种组合）+ dropless token-choice——这些在 DeepSeek 等旗舰上验证过的设计，被 OLMoE 缩到 6.9B/1.3B 激活尺寸做成**最小可复现配方**；论文消融显示"专家粒度 8→16→32→64 逐步带来收益"。
2. **辅助损失的有无是实验结论而非口号**：OLMoE **预训练明确使用负载均衡损失（0.01）+ router z-loss（0.001）**——论文实验显示没有负载均衡损失时专家利用率严重失衡（"死权重"），只有 SFT 阶段才去掉它。这与课件讲的 DeepSeek"无辅助损失路由"形成有意思的对照：**无辅助损失并非普适配方**，在小规模 MoE 上反而离不开。
3. **科学可复现（核心定位）**：与同规模 dense 模型在**相同数据、相同算力**下受控对比，论文结论是约 1.3B 激活的 MoE 显著优于同激活量 dense——MoE 的价值不只在旗舰尺度成立；权重、数据、代码、日志、中间检查点全开放。
4. **稳定性细节**：RMSNorm + QK-Norm（低精度训练更稳）、embedding 不共享且全部参数进 weight decay、无共享专家（论文实验显示共享专家在本模型无收益）。

**与课件已讲技术的对比**

课件讲了细粒度 MoE 与无辅助损失路由（DeepSeek 视角）；OLMoE 证明这些设计在**小模型上同样可以系统验证且完全可复现**——"MoE 不是旗舰专属，而是可以被科学研究的对象"。同时它给出一个重要反例：**在 1B 级 MoE 上，无辅助损失路由并不成立**，负载均衡损失是必需品（这与课件的旗舰结论形成规模相关的对照）。它还是"总参数 6.9B / 激活 1.3B"这种剪刀差的极端样本，与 Hunyuan-A13B（80B/13B）形成规模谱系的两端。

**亮点与局限**

- 亮点：开放度天花板（权重+数据+代码+日志）；约 1.3B 激活可单卡实验；与 dense 的严格受控对比、路由饱和/专家共激活等分析是方法论文献。
- 局限：上下文仅 4,096；**推理显存仍按总参数 6.9B 计**（激活参数低不等于内存省）；预训练 1% 时路由就已基本定型（路由饱和），后续数据多样性收益有限；Top-8 专家通信开销在小模型上占比不小；定位是研究工具而非产品旗舰。

**一句话结论**：OLMoE 把"细粒度 MoE"做成 6.9B/约 1.3B 激活、全开放的科研配方——MoE 从此可以被完整地复现和消融，也让"无辅助损失"的适用边界变得清晰。

---

## 五、支线 C：Dense 路线的结构级效率（不改稀疏性，改结构细节）

### 6. Gemma 2：logit cap + 交替局部/全局注意力

**模型概况**

| 项目 | 内容 |
|------|------|
| 发布 | Google DeepMind，9B/27B 于 2024-06-27、2B 于 2024-07-31（论文 *Gemma 2: Improving Open Language Models at a Practical Size*，arXiv:2408.00118） |
| 规模 | 2B / 9B / 27B，全部 **dense**（无 MoE）；层数 26 / 42 / 46，隐藏维度 2,304 / 3,584 / 4,608（官方论文 Table 1） |
| 注意力 | GQA：KV 头 4 / 8 / 16；**交替局部（滑窗 4,096）/ 全局（8,192）注意力**；query 预缩放 |
| 上下文 | 8,192（全局注意力跨度即 8,192） |
| 训练方式 | 2B/9B 用**知识蒸馏**训练（教师模型蒸馏），27B 从零预训练 |
| 结构关键词 | **logit soft-capping（注意力层 50.0 / 最终 logits 30.0）**、pre-norm + post-norm 双 RMSNorm、GeGLU、tied embedding（共享输入输出 embedding）、词表 256,128 |

**核心结构图解**

```
Gemma 2 层序列（交替模式）：

层1: 全局注意力（全序列 8,192）→ 层2: 局部注意力（滑窗 4,096）→ 层3: 全局 → …交替…

注意力 logits：logit soft-capping（50·tanh(x/50)）
最终输出 logits：logit soft-capping（30·tanh(x/30)）

attention 内部: GQA（KV 头 4/8/16）→ KV cache 更小
每个子层输入输出: 双 RMSNorm（pre-norm + post-norm）→ 深层训练更稳
```

**结构机制**

1. **logit soft-capping（最值得抄的细节）**：对注意力 logits（cap=50）和最终 logits（cap=30）做 `cap·tanh(x/cap)` 缩放，限制极端值——训练更稳定、允许更大学习率、缓解 logit 爆炸。这是一个**与稀疏化无关的通用结构技巧**。
2. **交替局部/全局注意力**：相邻层交替使用局部滑窗注意力（窗口 4,096）与全局注意力（跨度 8,192）——局部层大幅降低长序列计算/内存开销，全局层保证跨窗口信息流通，不引入检索精度损失。
3. **双 RMSNorm + GQA + 共享 embedding**：每个子层 pre-norm + post-norm 提升深层稳定性（26/42/46 层）；GQA 压缩 KV cache；输入输出 embedding 共享省参数。
4. **知识蒸馏训练（2B/9B）**：小尺寸版本用更大教师模型蒸馏训练而非纯 next-token 预测——"以小搏大"的关键训练侧结构决策。

**与课件已讲技术的对比**

课件把优化重心放在注意力压缩（MLA）与 MoE 上，dense 基本作为"基线"出现；Gemma 2 展示**纯 dense 也能靠结构细节提效**：logit cap 是训练稳定化、交替注意力是计算削减——两者都可以移植到任何模型（包括 MoE 模型）上，属于"结构无关的通用件"。

**亮点与局限**

- 亮点：结构简单、完全可复现；logit cap 被社区广泛借鉴；9B/27B 在开源 dense 中性价比突出；无 MoE 通信开销，部署简单；2B/9B 用蒸馏训练是"小模型借力大模型"的训练侧范本。
- 局限：上下文固定 8K（全局注意力跨度即 8K），长文档场景偏短；许可为 Gemma Terms of Use（非 OSI 开源许可，商用有条件限制）；蒸馏细节（教师模型、蒸馏权重）未完全公开；dense 的"参数-算力"比天然不如 MoE，27B 全量激活，长上下文推理成本线性上涨。

**一句话结论**：Gemma 2 证明"dense + 结构细节"仍是性价比极高的路线——不是所有模型都需要 MoE。

---

## 六、横向对比：六模型 × 六结构维度

| 维度 | Mamba-2 | RWKV-6 | Jamba | Grok-1 | OLMoE | Gemma 2 |
|------|---------|--------|-------|--------|-------|---------|
| 家族 | SSM（线性时间） | RNN 式线性注意力 | 混合（SSM+Attn+MoE） | 粗粒度 MoE | 细粒度 MoE | Dense |
| 总参数 | 最大 2.7B（官方） | 最大 14B | 52B（家族最大 398B） | 314B | 6.9B | 最大 27.2B |
| 激活参数 | 全部 | 全部 | **12B**（Large 为 32B） | 约 25%（估算，官方未公布） | **约 1.3B** | 全部 |
| 注意力层 | 无 | 无 | 1/8 层（SWA 窗口 4096） | GQA（48/8 头） | MHA（16 头）+QK-Norm | 交替局部(4K)/全局 |
| 上下文 | 4K 训练（理论无限） | 基座 4K / 微调 128K+ | **256K** | 8K | 4K | 8K |
| 训练/推理同构 | 否（并行扫描） | **是** | 否 | 否 | 否 | 否 |
| 检索/ICL 精度 | 弱（需混合） | 弱 | 中（attention 锚点兜底） | 强 | 强 | 强 |
| 开放度 | 权重+代码（Apache 2.0） | 权重+数据+代码（MIT） | v0.1 Apache 2.0 / 1.5 需申请 | 权重+代码（无训练细节） | **权重+数据+代码+日志** | 权重+代码（非 OSI 许可） |
| 一句话标签 | 统一理论 | 极简递推 | 生产混合 | 巨型粗粒度 | 可复现小 MoE | dense 效率化 |

## 七、结构演进启示（三条支线汇合）

1. **"要不要注意力"已成为第一性问题**：Mamba-2 给出统一理论（SSM≡结构化注意力）、RWKV-6 给出最激进的 RNN 实现、Jamba 给出生产折中——三者共同承认一个事实：**纯线性在精确检索上有短板，混合是当前的最优解**。这与课件里"线性注意力混合"的结论一致，但证据从"层内混合"扩展到了"整模型混合"。
2. **MoE 的粒度是一个连续谱**：Grok-1 的 8 专家 Top-2（粗）↔ OLMoE 的 64 专家 Top-8（细）↔ 课件里 DeepSeek 的 160+ 细粒度专家。粒度越小组合越灵活，但路由与通信开销越大——没有免费午餐，只有权衡。另一个规模相关发现：**"无辅助损失路由"在旗舰尺度可行（DeepSeek），但在 1B 级 MoE（OLMoE）上离不开负载均衡损失**——同一种设计在不同规模下的适用性不同。
3. **Dense 没有过时，且贡献了"通用件"**：Gemma 2 的 logit cap、交替注意力与双层 RMSNorm 不依赖稀疏性，可以被任何架构白嫖——结构细节的收益与"是否 MoE"正交。
4. **评估结构不能只看总参数**：至少同时记录激活参数、上下文长度、精确检索能力、训练/推理是否同构、是否需要专用 kernel——本作业对比表的六列维度正是这个口径。

## 八、总结

本周调研了 6 个课上没讲过的开源模型，覆盖 3 条结构支线：

1. **线性时间架构**：Mamba-2（SSD 统一理论）、RWKV-6（RNN 极简实现）、Jamba（混合生产配方）——回答"注意力能否被替换"；
2. **MoE 两种开源配方**：Grok-1（314B 粗粒度全量开源）、OLMoE（6.9B/约 1.3B 激活全开放科研配方）——回答"稀疏激活的两个极端长什么样"；
3. **Dense 结构效率**：Gemma 2（logit cap + 交替注意力）——回答"不改稀疏性还能怎么提效"。


**方法论收获**：结构调研的价值不在罗列参数，而在"**与已讲内容的差异点**"——每个模型只回答一个课件没回答的结构问题。

## 九、参考资料

1. Mamba-2 论文（SSD）：*Transformers are SSMs*，arXiv:2405.21060 — https://arxiv.org/abs/2405.21060 ；官方代码：https://github.com/state-spaces/mamba ；普林斯顿 PLI 博客（Mamba-2: Algorithms and Systems）：https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems ；官方模型卡：https://huggingface.co/state-spaces/mamba2-2.7b
2. RWKV-6 论文：*Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence*，arXiv:2404.05892 — https://arxiv.org/abs/2404.05892 ；官方博客（Finch 14B）：https://blog.rwkv.com/p/rwkv-v6-finch-14b-is-here ；代码：https://github.com/BlinkDL/RWKV-LM ；官方模型卡：https://huggingface.co/RWKV/rwkv-6-world-7b
3. Jamba 论文：*Jamba: A Hybrid Transformer-Mamba Language Model*，arXiv:2403.19887 — https://arxiv.org/abs/2403.19887 ；Jamba-1.5 论文：arXiv:2408.12570 — https://arxiv.org/abs/2408.12570 ；模型卡：https://huggingface.co/ai21labs/Jamba-v0.1 ；AI21 官方博客：https://www.ai21.com/blog/announcing-jamba-model-family/ ；Jamba-1.5 介绍（NVIDIA）：https://developer.nvidia.cn/blog/jamba-1-5-llms-leverage-hybrid-architecture-to-deliver-superior-reasoning-and-long-context-handling/
4. Grok-1 官方公告（xAI）：https://x.ai/news/grok-os ；官方仓库（无论文，仅有推理代码）：https://github.com/xai-org/grok-1 ；架构说明（DeepWiki）：https://deepwiki.com/xai-org/grok-1/3-model-architecture ；权重：https://huggingface.co/xai-org/grok-1
5. OLMoE 论文：*OLMoE: Open Mixture-of-Experts Language Models*，arXiv:2409.02060 — https://arxiv.org/abs/2409.02060 ；AI2 官方博客：https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model ；代码：https://github.com/allenai/OLMoE ；官方模型卡（含 config.json）：https://huggingface.co/allenai/OLMoE-1B-7B-0924
6. Gemma 2 论文：*Gemma 2: Improving Open Language Models at a Practical Size*，arXiv:2408.00118 — https://arxiv.org/abs/2408.00118 ；Google 官方技术博客：https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/ ；HF 发布博客：https://huggingface.co/blog/gemma2 ；模型卡：https://huggingface.co/google/gemma-2-27b

> 注：文中所有结构参数均以官方论文/官方仓库/官方 config.json/官方博客公开信息为准；Grok-1 的激活参数为按 8 专家 Top-2 的估算（官方未明确公布），且其训练细节（如是否使用辅助负载均衡损失）官方未披露；Jamba-1.5 系列细节以 AI21 官方发布为准；OLMoE 的"无辅助损失"仅指其 SFT 阶段（预训练使用负载均衡损失 + router z-loss）。
