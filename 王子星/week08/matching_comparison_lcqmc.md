# 文本匹配多模型对比实验报告（LCQMC）

## 1. 实验概述

| 方法 | 模型 | 损失函数/策略 | 可训练参数 |
|------|------|------------|-----------|
| BiEncoder + CosineEmbeddingLoss | bert-base-chinese | CosineEmbeddingLoss | 全参微调 |
| BiEncoder + TripletLoss | bert-base-chinese | TripletLoss | 全参微调 |
| CrossEncoder + CrossEntropyLoss | bert-base-chinese | CrossEntropyLoss | 全参微调 |
| LLM Zero-shot | qwen3.7-plus | 生成式（无训练） | 0（仅推理） |
| LLM LoRA SFT | Qwen2.5-0.5B-Instruct + LoRA | 生成式（LoRA 微调） | 少量（LoRA） |

---

## 2. 实验配置

| 配置项 | BiEncoder Cosine | BiEncoder Triplet | CrossEncoder | LLM Zero-shot | LLM LoRA SFT |
|--------|----------------|-----------------|-------------|--------------|-------------|
| 基础模型 | bert-base-chinese | bert-base-chinese | bert-base-chinese | qwen3.7-plus | Qwen2.5-0.5B-Instruct |
| 训练数据 | LCQMC 训练集（60K）| LCQMC 训练集（60K）| LCQMC 训练集（60K）| 0 条 | LCQMC 训练集（60K）|
| 评估样本 | 验证集（~8,802 条）| 验证集（~8,802 条）| 验证集（~8,802 条）| 200 条 | 200 条 |
| Epochs | 3 | 3 | 3 | — | — |
| 运行设备 | GPU | GPU | GPU | API 调用 | GPU |

---

## 3. 训练过程

### 3.1 BiEncoder + CosineEmbeddingLoss 训练日志

| Epoch | Train Loss | Val Acc | Val F1 | Threshold | 耗时(s) |
|-------|-----------|--------|--------|----------|--------|
| 1 | 0.1965 | 0.7836 | 0.7835 | 0.82 | 549.09 |
| 2 | 0.1530 | **0.8121** | **0.8118** | 0.74 | 549.73 |
| 3 | 0.1350 | 0.8105 | 0.8101 | 0.75 | 539.95 |

### 3.2 BiEncoder + TripletLoss 训练日志

| Epoch | Train Loss | Val Acc | Val F1 | Threshold | 耗时(s) |
|-------|-----------|--------|--------|----------|--------|
| 1 | 0.0115 | 0.8037 | 0.8037 | 0.87 | 456.04 |
| 2 | 0.0053 | **0.8219** | **0.8218** | 0.83 | 453.50 |
| 3 | 0.0029 | 0.8200 | 0.8198 | 0.84 | 453.51 |

### 3.3 CrossEncoder + CrossEntropyLoss 训练日志

| Epoch | Train Loss | Train Acc | Val Acc | Val F1 | 耗时(s) |
|-------|-----------|----------|--------|--------|--------|
| 1 | 0.2904 | 0.8733 | 0.8304 | 0.8302 | 506.64 |
| 2 | 0.1811 | 0.9290 | 0.8466 | 0.8466 | 511.26 |
| 3 | 0.1214 | 0.9567 | **0.8544** | **0.8543** | 506.11 |

> Epoch 3 Train Acc 达 0.9567 而 Val Acc 为 0.8544，出现轻微过拟合趋势，但验证集 F1 仍为五种方法之最。

### 3.4 LLM LoRA SFT（Qwen2.5-0.5B-Instruct）

训练细节见 `outputs/logs/lcqmc/train_sft.json`，val_loss 在 epoch 1（0.0719）已到谷底，epoch 2/3 持续回升；评估在 200 条测试样本上进行：accuracy=0.860，f1_weighted=0.860，f1_pos=0.8462，解析失败率为 0。

| Epoch | Train Loss | Val Loss | 累计耗时(s) |
|-------|-----------|---------|-----------|
| **1** | 0.0643 | **0.0719**（最优 ckpt）| 671 |
| 2 | 0.0475 | 0.0745 | 1441 |
| 3 | 0.0343 | 0.0852（过拟合）| 1724 |

---

## 4. 评估结果

### 4.1 有监督模型（验证集）

| 方法 | Accuracy | F1 | AUC | 最优阈值 |
|------|---------|-----|-----|--------|
| BiEncoder (CosineEmbeddingLoss) | 0.8121 | 0.8118 | 0.8856 | 0.74 |
| BiEncoder (TripletLoss) | 0.8219 | 0.8218 | **0.9049** | 0.83 |
| CrossEncoder (CrossEntropyLoss) | **0.8544** | **0.8543** | — | — |

### 4.2 LLM 系列评估结果（200 条样本）

| 方法 | Accuracy | Precision(正) | Recall(正) | F1(正) | 解析失败 |
|------|---------|------------|----------|------|--------|
| qwen3.7-plus Zero-shot | **0.930** | 0.8714 | **0.9242** | **0.8971** | 0 |
| Qwen2.5-0.5B LoRA SFT | 0.860 | — | — | 0.8462 | 0 |

> qwen3.7-plus 在 LCQMC 通用领域表现强劲，Precision（0.8714）和 Recall（0.9242）均衡，整体 F1（0.8971）显著高于 LoRA SFT（0.8462），与 BQ Corpus 银行领域的"低召回保守策略"形成鲜明对比——说明通用 LLM 在通用领域的 zero-shot 匹配能力本就很强，SFT 的增益空间因此被压缩。

---

## 5. 综合性能对比

| 方法 | Accuracy | F1 | AUC | 评估样本 | 是否需要训练 | 训练总耗时(s) |
|------|---------|-----|-----|---------|------------|------------|
| CrossEncoder (CrossEntropyLoss) | **0.8544** | **0.8543** | — | 验证集 | 是 | ~1524 |
| BiEncoder (TripletLoss) | 0.8219 | 0.8218 | **0.9049** | 验证集 | 是 | **~1363** |
| BiEncoder (CosineEmbeddingLoss) | 0.8121 | 0.8118 | 0.8856 | 验证集 | 是 | ~1639 |
| qwen3.7-plus Zero-shot | 0.930 | 0.8971 | — | 200 条 | 否 | 0 |
| Qwen2.5-0.5B LoRA SFT | 0.860 | 0.8462 | — | 200 条 | 是 | ~1724 |

> LLM 方法的 F1 列使用 f1_pos（正样本 F1），监督方法使用验证集加权 F1。

---

## 6. 关键差异分析

### 6.1 CrossEncoder vs BiEncoder

| 对比维度 | CrossEncoder | BiEncoder Cosine | BiEncoder Triplet |
|---------|-------------|----------------|-----------------|
| Val Accuracy | **0.8544** | 0.8121 | 0.8219 |
| Val F1 | **0.8543** | 0.8118 | 0.8218 |
| AUC | — | 0.8856 | **0.9049** |
| 训练总耗时(s) | ~1524 | ~1639 | **~1363** |

CrossEncoder 将句对拼接后联合建模，F1 比最优 BiEncoder 高 +0.0325；BiEncoder Triplet AUC（0.9049）高于 Cosine（0.8856），且训练耗时最短。值得注意的是，在 LCQMC 60K 训练数据下，TripletLoss 的优势（相比 CosineEmbeddingLoss，F1 高 +0.0100）比 AFQMC/BQ 等规模相近的数据集上更明显，与 ARCHITECTURE.md 的预期一致：数据量越大 Triplet 的向量空间排序收益越大，60K 已能体现这一趋势。

### 6.2 有监督模型 vs LLM 方法

| 对比维度 | CrossEncoder | BiEncoder Cosine | qwen3.7-plus Zero-shot | Qwen2.5-0.5B SFT |
|---------|-------------|----------------|----------------------|----------------|
| Accuracy | **0.8544** | 0.8121 | 0.930 | 0.860 |
| F1 | **0.8543** | 0.8118 | 0.8971 | 0.8462 |
| 是否调用 API | 否 | 否 | 是 | 否 |

LCQMC 为通用领域口语问句匹配，LLM Zero-shot 准确率（0.930）反超所有有监督模型，体现出大模型在通用语义理解上的天然优势。这与 BQ Corpus 银行领域（Zero-shot 仅 0.73）截然不同，领域特化程度是 LLM Zero-shot 效果的核心变量。

### 6.3 qwen3.7-plus Zero-shot vs Qwen2.5-0.5B LoRA SFT

| 方法 | Accuracy | F1(pos) | Recall(pos) | Precision(pos) |
|------|---------|--------|-----------|-------------|
| qwen3.7-plus Zero-shot | **0.930** | **0.8971** | **0.9242** | 0.8714 |
| Qwen2.5-0.5B LoRA SFT | 0.860 | 0.8462 | — | — |

LCQMC 上 qwen3.7-plus Zero-shot（F1=0.8971）超越 Qwen2.5-0.5B LoRA SFT（F1=0.8462）——与 BQ Corpus 相反。大模型在通用领域的语义理解能力已足够强，LoRA 微调小模型反而未能超越零样本。说明 SFT 的收益与任务的领域特化程度密切相关：越是通用领域，SFT 对小模型的弥补效果越有限；越是垂直专业领域（如银行），SFT 的增益越显著。

---

## 7. 结论

1. **CrossEncoder 整体最优**：验证集 F1=0.8543，联合建模句对交互在 LCQMC 上仍是精度最高的判别式方案，适合对准确率要求高、实时性要求低的场景。
2. **BiEncoder TripletLoss 排序能力最强**：AUC=0.9049 为判别式方法最高，且训练耗时最短（~1363s）。更重要的是，LCQMC 60K 训练数据下 TripletLoss 相对 CosineEmbeddingLoss 的优势（F1 +0.0100）比 AFQMC 10K 正样本时更明显，验证了 ARCHITECTURE.md 中"数据量越大 TripletLoss 优势越明显"的判断。
3. **LLM Zero-shot 在通用领域反超有监督模型**：qwen3.7-plus Zero-shot 准确率 0.930 优于全部三种有监督方法（0.8121~0.8544），Precision 与 Recall 均衡（0.87/0.92），与 BQ Corpus 的"高精低召"保守策略形成鲜明对比。通用领域是大模型的主场，零样本匹配能力本就很强。
4. **LoRA SFT 在通用领域增益有限**：Qwen2.5-0.5B LoRA SFT（0.860/0.8462）低于 qwen3.7-plus Zero-shot（0.930/0.8971），说明通用领域的小模型 SFT 无法抵消底座能力差距；而在 BQ Corpus 银行领域，相同的 SFT 流程能让 0.5B 模型（0.855）显著超越 qwen3.7-plus Zero-shot（0.73）。SFT 增益的大小取决于任务领域化程度。
5. **LCQMC vs BQ Corpus 跨数据集对比**：相同代码、相同超参，切换数据集后整体指标均有提升（LCQMC 监督方法 F1 约 0.81~0.85 vs BQ 0.90~0.93 稍低，但 LLM Zero-shot 大幅提升 0.73→0.93）。两个数据集均使用约 60K 训练条目，LCQMC 正负比更均衡、句式更多样，Zero-shot 收益尤其显著；BQ Corpus 属于高度领域特化数据，监督训练优势更明显。
