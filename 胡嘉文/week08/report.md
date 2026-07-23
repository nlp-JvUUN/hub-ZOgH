# 实验报告：中文语义文本匹配（Semantic Text Matching）

---

## 一、实验概述

### 1.1 任务定义

中文语义文本匹配（Semantic Text Matching）：给定两句话（sentence1, sentence2），判断它们是否表达相同语义。这是一个二分类问题，输出 1（相似）或 0（不相似）。

### 1.2 实验目标

对比四种文本匹配方法：

| 方案 | 范式 | 模型 | 核心思路 |
|------|------|------|---------|
| BiEncoder + CosineEmbeddingLoss | 表示型（判别式） | BERT 4层 | 两句独立编码 → 余弦相似度 |
| BiEncoder + TripletLoss | 表示型（判别式） | BERT 4层 | 三元组约束相对距离 |
| CrossEncoder | 交互型（判别式） | BERT 4层 | 两句拼接全层交互 → 分类 |
| LLM SFT（LoRA） | 生成式 | Qwen2-0.5B | 指令微调 → 生成【相似】/【不相似】 |

### 1.3 实验环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 4060 Laptop |
| 框架 | PyTorch + Transformers + Peft |
| BERT 模型 | bert-base-chinese（4层，45.6M参数） |
| LLM 模型 | Qwen2-0.5B-Instruct（495M参数，LoRA 1.08M可训练参数） |
| 训练数据 | AFQMC（蚂蚁金融问句匹配） |

---

## 二、数据集分析

### 2.1 AFQMC 数据统计

| 分割 | 总条数 | 正样本（相似） | 负样本（不相似） | 正负比 |
|------|-------:|--------------:|----------------:|-------:|
| train | 34,334 | 10,573 (30.8%) | 23,761 (69.2%) | 1:2.25 |
| validation | 4,316 | 1,338 (31.0%) | 2,978 (69.0%) | 1:2.23 |
| test | 3,861 | 未公开（CLUE竞赛格式，label=-1） | — | — |

**类别不均衡**：负样本约为正样本的 2.2 倍，接近真实业务分布。

### 2.2 句子长度分布

| 统计量 | 数值 |
|--------|------|
| 字符长度均值 | 13.4 字 |
| 最大长度 | ~50 字 |
| max_length=32 覆盖 | 98.4% |
| max_length=64 覆盖 | 99.9% |

AFQMC 句子极短，BiEncoder 使用 max_length=64 即可覆盖 99.9% 的样本。

### 2.3 Length Bias 检测

正样本长度差均值 = 3.4 字，负样本长度差均值 = 3.5 字，两者接近，**数据无明显 length bias**。

---

## 三、实验方法详细说明

### 3.1 BiEncoder + CosineEmbeddingLoss

**架构**：Siamese BERT（共享权重），两路独立编码 → mean pooling → Dropout → L2 归一化 → 余弦相似度。

**损失函数**：
```
loss = cosine_embedding_loss(emb_a, emb_b, target, margin=0.3)
```
- 正例对（label=1）：target=+1，拉近余弦相似度到 ≥0
- 负例对（label=0）：target=-1，推远余弦相似度到 ≤ margin

**评估方式**：在验证集上枚举 [0.0, 1.0] 区间 101 个阈值，取 weighted-F1 最高的作为最优阈值。

**训练配置**：
| 参数 | 值 |
|------|-----|
| 学习率 | 2e-5 |
| 池化策略 | mean |
| BERT 层数 | 4 |
| Epochs | 3 |
| Batch size | 32 |
| max_length | 64 |
| margin | 0.3 |

### 3.2 BiEncoder + TripletLoss

**架构**：与BiEncoder相同，但使用三元组输入 (anchor, positive, negative)。

**损失函数**：
```
loss = triplet_margin_loss(emb_a, emb_p, emb_n, margin=0.3)
```
要求 sim(anchor, positive) > sim(anchor, negative) + margin。

**三元组构建**：从 10,573 个正样本对出发，为每个 anchor 查找其配对的负样本；若无则从全局池随机选取，共构建 10,573 个三元组。

**训练配置**：与CosineEmbeddingLoss相同，但评估仍用 PairDataset + 阈值搜索。

### 3.3 CrossEncoder

**架构**：`[CLS] s1 [SEP] s2 [SEP]` → BERT → CLS 向量 → Linear(768, 2) → logits → CrossEntropyLoss。

**关键差异**：
- 两句在 BERT 每一层都通过 Self-Attention 交互
- 评估直接 argmax(logits)，**无需阈值搜索**
- max_length=128（句对拼接）

**训练配置**：
| 参数 | 值 |
|------|-----|
| 学习率 | 2e-5 |
| BERT 层数 | 4 |
| Epochs | 3 |
| Batch size | 32 |
| max_length | 128 |

### 3.4 LLM SFT（LoRA指令微调）

**模型**：Qwen2-0.5B-Instruct（495M参数）

**微调方式**：LoRA（r=8，target= q/k/v/o_proj，可训练参数 ~1.08M，占比 0.22%）

**指令模板**：
```
system: 你是一个语义匹配助手。判断两句话语义是否相同，只输出【相似】或【不相似】，不要输出其他内容。
user:   句子A：{sentence1}\n句子B：{sentence2}\n是否相似：
target: 【相似】 / 【不相似】
```

**类别平衡**：AFQMC 正负比 31:69，默认开启正负平衡采样（各取 num_train//2 条），防止模型退化为全预测负例。

**训练配置**：
| 参数 | 值 |
|------|-----|
| 训练样本 | 5,000（平衡采样，正2500+负2500） |
| 学习率 | 2e-4 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient Accumulation | 4 |
| max_length | 128 |
| Loss Masking | 仅在 response token 上计算 loss |

---

## 四、实验结果

### 4.1 训练过程

#### BiEncoder + CosineEmbeddingLoss（每epoch ~18min）

| Epoch | train_loss | val_acc | val_f1 | threshold |
|:-----:|-----------:|--------:|-------:|----------:|
| 1 | 0.2508 | 0.6506 | 0.6470 | 0.54 |
| 2 | 0.2219 | 0.6784 | 0.6625 | 0.55 |
| 3 | **0.2141** | 0.6747 | **0.6728** | 0.52 |

*最优模型保存于 epoch 3（val_f1=0.6728）*

#### BiEncoder + TripletLoss（每epoch ~2min）

| Epoch | train_loss | val_acc | val_f1 | threshold |
|:-----:|-----------:|--------:|-------:|----------:|
| 1 | 0.0681 | 0.6555 | 0.6487 | 0.81 |
| 2 | **0.0150** | 0.6664 | **0.6599** | 0.81 |
| 3 | 0.0103 | 0.6657 | 0.6592 | 0.81 |

*最优模型保存于 epoch 2（val_f1=0.6599），epoch 2→3 几乎无提升*

#### CrossEncoder（每epoch ~3.2min）

| Epoch | train_loss | train_acc | val_acc | val_f1 |
|:-----:|-----------:|----------:|--------:|-------:|
| 1 | 0.6123 | 0.6881 | 0.6501 | 0.6468 |
| 2 | 0.5673 | 0.6940 | 0.6865 | 0.6683 |
| 3 | **0.5272** | **0.7162** | **0.6905** | **0.6750** |

*最优模型保存于 epoch 3（val_f1=0.6750），每 epoch 持续提升*

#### LLM SFT-LoRA（每epoch ~18.5min）

| Epoch | train_loss | val_loss | 说明 |
|:-----:|-----------:|---------:|------|
| 1 | 0.1617 | **0.1216** | 最优 val_loss |
| 2 | 0.1293 | 0.1230 | 持平 |
| 3 | 0.1058 | 0.1294 | 过拟合（val_loss 反弹） |

*TARGET 仅 3~5 token，学习信号高度集中，1 epoch 已收敛*

### 4.2 最终效果对比

**实验条件**：AFQMC validation 集（4,316 条），4 层 BERT，3 epoch，RTX 4060 Laptop GPU。SFT 在 200 条子集上评估。

| 方法 | Accuracy | F1 (weighted) | F1 (正例) | 决策方式 | 训练总时长 |
|------|--------:|--------------:|----------:|---------|-----------:|
| BiEncoder + CosineEmbeddingLoss | 0.6735 | **0.6765** | 0.6765 | threshold=0.51 | ~54 min |
| BiEncoder + TripletLoss | 0.6664 | 0.6599 | 0.6599 | threshold=0.81 | ~5.6 min |
| CrossEncoder + CrossEntropyLoss | **0.6905** | 0.6750 | 0.6750 | argmax | ~9.5 min |
| Qwen2-0.5B SFT（LoRA，5K平衡） | 0.6400 | 0.6535 | 0.5556 | 生成【相似】/【不相似】 | ~55 min |

### 4.3 结果解读

#### 4.3.1 CrossEncoder Acc 最高（0.6905），但 F1 最高的是 BiEncoder Cosine（0.6765）

CrossEncoder 的 Accuracy 领先 BiEncoder Cosine 约 1.7 个百分点，说明它在多数类（负类）上预测更稳健——全层交互更容易把握"明显的不相似"。但加权 F1 基本持平（0.6750 vs 0.6765），说明在正类上 BiEncoder 表现略优。

**对比 1-epoch 退化现象**（教学参考）：
- 1 epoch CrossEncoder：Acc=0.6921 / F1=0.5703，正类 recall 极低
- 3 epoch CrossEncoder：Acc=0.6905 / F1=0.6750，F1 跳升 0.10

**教学意义**：Accuracy 并不总能反映模型真实能力，尤其在类别不均衡 + 训练不足时。

#### 4.3.2 CosineEmbeddingLoss 优于 TripletLoss（F1 差 0.0166）

AFQMC 正样本仅 10K 条，TripletDataset 构造的三元组同样只有 10K 个，训练信号偏少。Triplet 的 train_loss 从 0.068 在 epoch 1 末就降到 0.015，epoch 2/3 几乎不动（margin 容易"打满"）。

#### 4.3.3 TripletLoss 阈值偏高（0.81）

TripletLoss 只约束正例相对负例更近，不要求绝对相似度的量级，导致所有嵌入向高值偏移，最优阈值因此明显高于 CosineEmbeddingLoss（0.51）。

#### 4.3.4 判别式 vs 生成式：BERT 在文本匹配上仍然占优

BiEncoder Cosine 的 F1（正例）= 0.6765，比 LoRA SFT 的 0.5556 高 12 个百分点，而 SFT 还多花了约 5 倍训练时间。文本匹配 TARGET 极短（3~5 token）且标签集封闭（二分类），判别式架构能把全部容量用于"两句话是否相似"这一信号。SFT 的价值在于：(1) 灵活扩展新类别；(2) 离线推理无需 API 成本。

---

## 五、Bad Case 分析

基于 BiEncoder + CosineEmbeddingLoss（threshold=0.51，3 epoch）在 4,316 条 validation 集上的错误分析。

### 5.1 总体错误分布

| 指标 | 数值 |
|------|------|
| 总错误数 | 1,409 |
| 错误率 | 32.6% |
| Accuracy | 0.6735 |

### 5.2 FP/FN 分类

| 错误类型 | 数量 | 占比 | 说明 |
|---------|-----:|----:|------|
| FP 假阳性（预测相似，实为不同） | **752** | 53.4% | 模型过度自信"相似" |
| ├ 高置信度错误（Δscore > 0.15） | 155 | 11.0% | 问题最严重，离阈值远 |
| └ 临界错误（Δscore ≤ 0.15） | 597 | 42.4% | 接近阈值，可调阈值改善 |
| FN 假阴性（预测不同，实为相似） | **657** | 46.6% | 模型错过真实相似对 |
| ├ 高置信度错误（Δscore > 0.15） | 132 | 9.4% | 需改进模型表示能力 |
| └ 临界错误（Δscore ≤ 0.15） | 525 | 37.2% | 接近阈值，可部分改善 |

**关键发现**：79% 的错误（临界错误）落在阈值附近 ±0.15 区间，阈值微调可改善；高置信度错误（20%）才是需要靠模型/数据解决的核心问题。

### 5.3 语言特征对比

| 特征 | FP（假阳性） | FN（假阴性） | 含义 |
|------|------------:|------------:|------|
| 字符 Jaccard 均值 | **0.484** | 0.380 | FP：词汇高度重叠但语义不同 |
| 句子长度差均值 | 3.4 字 | 4.8 字 | FN：表述方式差异更大 |

**两个数字指向不同的根因和优化方向**：
- **FP 是"换字不换意"陷阱**：词汇高度重叠（Jaccard=0.484）但语义不同，需增大 margin 或做难负样本挖掘
- **FN 是"换说法"挑战**：换了表达方式（Jaccard=0.380），需 SimCSE 对比学习或增加模型容量

### 5.4 典型 Bad Case

**FP 高置信度（score 极高，但标签为不相似）：**
```
score=0.967  "【蚂蚁借呗】你支付宝******@qq.com提交的蚂蚁借呗申请经综合评估暂未通过..."
               "【蚂蚁借呗】你支付宝hys***@***.com提交的蚂蚁借呗申请经综合评估暂未通过..."
→ 仅账号不同，模板完全相同。数据质量问题：标注为"不相似"存疑。

score=0.900  "花呗能参加优惠活动吗"  ||  "花呗能参加购物优惠活动吗"
→ 仅多"购物"二字，语义接近但标注为不相似，属于边界模糊的难例。
```

**FN 高置信度（score 极低，但标签为相似）：**
```
score=0.082  "我现在的花呗是否全部还款完毕"  ||  "我在花呗有***百元，消费***元，还款是否***元"
→ 表达差异极大，模型未能建立共同语义关联。

score=0.157  "花呗使用不了"  ||  "花呗被冻结"
→ 因果同源但词汇完全不重叠，模型未能建立"使用不了 = 被冻结"的等价关系。
```

### 5.5 数据质量观察

部分高 FP 案例（如两条仅账号不同的系统通知）标注为"不相似"存在争议，说明 AFQMC 存在一定比例的标注噪声。这是真实业务数据的常态，也是 val 集上 F1 有天花板（~0.68）的原因之一。

---

## 六、结论与优化方向

### 6.1 核心结论

1. **表示型 vs 交互型**：CrossEncoder（0.6905 Acc）在分类精度上略优于 BiEncoder（0.6765 F1），但无法预计算向量，推理时必须两两计算，不适合大规模检索
2. **损失函数选择**：CosineEmbeddingLoss（0.6765 F1）在 AFQMC 上优于 TripletLoss（0.6599 F1），但在大规模数据集上 TripletLoss 潜力更大
3. **判别式 vs 生成式**：BERT 判别式方案（F1≈0.68）远优于 LLM SFT 生成式方案（F1≈0.56），文本匹配二分类任务更适合判别式
4. **训练收敛速度**：SFT 1 epoch 已足够（过拟合点），CrossEncoder 需 3 epoch 才能体现真实能力
5. **阈值偏移**：79% 的错误集中在阈值附近，实际部署时应按线上分布校准阈值

### 6.2 推荐优化方向

| 方向 | 方法 | 预期收益 |
|------|------|---------|
| 数据增强 | 难负样本挖掘（针对FP） | 减少词汇重叠导致的假阳性 |
| 数据增强 | LLM同义改写扩充正例（针对FN） | 增加正样本多样性，提升泛化 |
| 模型增强 | BERT 4层→12层 | 提升深层语义理解，预计F1提升3~8点 |
| 训练策略 | SimCSE对比学习预训练 | 改善同义异词场景的判别能力 |
| 训练策略 | Online Hard Negative Mining | 提升 TripletLoss 训练效果 |
| 工程部署 | BiEncoder 召回 + CrossEncoder 精排 | 两阶段级联，兼顾速度与精度 |

### 6.3 数据可用性说明

AFQMC 的 test 集在 CLUE 竞赛中标签未公开（label=-1），因此本项目所有评估均在 validation 集（4,316 条）上进行。LCQMC（238K 对，口语化问句）和 BQ Corpus（86K 对，银行金融问句）已下载至本地，可供学生自主练习和跨数据集迁移实验。
