# 文本匹配多模型对比实验报告（BQ Corpus）

## 1. 实验概述

| 方法                            | 模型                         | 损失函数/策略       | 可训练参数   |
| ------------------------------- | ---------------------------- | ------------------- | ------------ |
| BiEncoder + CosineEmbeddingLoss | bert-base-chinese            | CosineEmbeddingLoss | 全参微调     |
| BiEncoder + TripletLoss         | bert-base-chinese            | TripletLoss         | 全参微调     |
| CrossEncoder + CrossEntropyLoss | bert-base-chinese            | CrossEntropyLoss    | 全参微调     |
| LLM Zero-shot                   | qwen3.7-plus                 | 生成式（无训练）    | 0（仅推理）  |
| LLM LoRA SFT                    | Qwen2.5-0.5B-Instruct + LoRA | 生成式（LoRA 微调） | 少量（LoRA） |

---

## 2. 实验配置

| 配置项   | BiEncoder Cosine  | BiEncoder Triplet | CrossEncoder      | LLM Zero-shot | LLM LoRA SFT          |
| -------- | ----------------- | ----------------- | ----------------- | ------------- | --------------------- |
| 基础模型 | bert-base-chinese | bert-base-chinese | bert-base-chinese | qwen3.7-plus  | Qwen2.5-0.5B-Instruct |
| 训练数据 | BQ Corpus 训练集  | BQ Corpus 训练集  | BQ Corpus 训练集  | 0 条          | BQ Corpus 训练集      |
| 评估样本 | 验证集            | 验证集            | 验证集            | 200 条        | 200 条                |
| Epochs   | 3                 | 3                 | 3                 | —             | —                     |
| 运行设备 | GPU               | GPU               | GPU               | API 调用      | GPU                   |

---

## 3. 训练过程

### 3.1 BiEncoder + CosineEmbeddingLoss 训练日志

| Epoch | Train Loss | Val Acc    | Val F1     | Threshold | 耗时(s) |
| ----- | ---------- | ---------- | ---------- | --------- | ------- |
| 1     | 0.2047     | 0.8515     | 0.8514     | 0.71      | 657.74  |
| 2     | 0.1401     | 0.8912     | 0.8911     | 0.69      | 669.83  |
| 3     | 0.1173     | **0.9005** | **0.9004** | 0.72      | 674.14  |

### 3.2 BiEncoder + TripletLoss 训练日志

| Epoch | Train Loss | Val Acc    | Val F1     | Threshold | 耗时(s) |
| ----- | ---------- | ---------- | ---------- | --------- | ------- |
| 1     | 0.0895     | 0.8587     | 0.8587     | 0.58      | 477.44  |
| 2     | 0.0266     | 0.8881     | 0.8881     | 0.57      | 476.04  |
| 3     | 0.0131     | **0.8958** | **0.8958** | 0.54      | 485.05  |

### 3.3 CrossEncoder + CrossEntropyLoss 训练日志

| Epoch | Train Loss | Train Acc | Val Acc    | Val F1     | 耗时(s) |
| ----- | ---------- | --------- | ---------- | ---------- | ------- |
| 1     | 0.4101     | 0.8116    | 0.8877     | 0.8877     | 634.41  |
| 2     | 0.2195     | 0.9149    | 0.9182     | 0.9182     | 625.65  |
| 3     | 0.1231     | 0.9577    | **0.9265** | **0.9265** | 611.53  |

> Epoch 3 Train Acc 达 0.9577 而 Val Acc 为 0.9265，出现轻微过拟合趋势，但验证集 F1 仍为五种方法之最。

### 3.4 LLM LoRA SFT（Qwen2.5-0.5B-Instruct）

训练细节见 `outputs/bq_corpus/logs/train_sft.json`，评估在 200 条测试样本上进行：accuracy=0.855，f1_weighted=0.8547，f1_pos=0.8722，解析失败率为 0。

---

## 4. 评估结果

### 4.1 有监督模型（验证集）

| 方法                            | Accuracy   | F1         | AUC        | 最优阈值 |
| ------------------------------- | ---------- | ---------- | ---------- | -------- |
| BiEncoder (CosineEmbeddingLoss) | 0.9005     | 0.9004     | 0.9446     | 0.72     |
| BiEncoder (TripletLoss)         | 0.8958     | 0.8958     | **0.9590** | 0.54     |
| CrossEncoder (CrossEntropyLoss) | **0.9265** | **0.9265** | —          | —        |

### 4.2 LLM 系列评估结果（200 条样本）

| 方法                   | Accuracy  | Precision(正) | Recall(正) | F1(正)     | 解析失败 |
| ---------------------- | --------- | ------------- | ---------- | ---------- | -------- |
| qwen3.7-plus Zero-shot | 0.73      | **0.8000**    | 0.2424     | 0.3721     | 0        |
| Qwen2.5-0.5B LoRA SFT  | **0.855** | —             | —          | **0.8722** | 0        |

> qwen3.7-plus 精确率高（0.8000）但召回率极低（0.2424），整体为保守预测策略；LoRA SFT 准确率（0.855）和 F1（0.8722）显著高于 Zero-shot，说明在 BQ Corpus 银行领域任务上，经过 LoRA 微调的 Qwen2.5-0.5B 效果大幅提升。

---

## 5. 综合性能对比

| 方法                            | Accuracy   | F1         | AUC        | 评估样本 | 是否需要训练 | 训练总耗时(s) |
| ------------------------------- | ---------- | ---------- | ---------- | -------- | ------------ | ------------- |
| CrossEncoder (CrossEntropyLoss) | **0.9265** | **0.9265** | —          | 验证集   | 是           | ~1871         |
| BiEncoder (CosineEmbeddingLoss) | 0.9005     | 0.9004     | 0.9446     | 验证集   | 是           | ~2001         |
| BiEncoder (TripletLoss)         | 0.8958     | 0.8958     | **0.9590** | 验证集   | 是           | **~1438**     |
| qwen3.7-plus Zero-shot          | 0.73       | 0.3721     | —          | 100 条   | 否           | 0             |
| Qwen2.5-0.5B LoRA SFT           | 0.855      | 0.8547     | —          | 200 条   | 是           | —             |

> LLM 方法的 F1 列使用 f1_pos（正样本 F1），监督方法使用验证集加权 F1。

---

## 6. 关键差异分析

### 6.1 CrossEncoder vs BiEncoder

| 对比维度      | CrossEncoder | BiEncoder Cosine | BiEncoder Triplet |
| ------------- | ------------ | ---------------- | ----------------- |
| Val Accuracy  | **0.9265**   | 0.9005           | 0.8958            |
| Val F1        | **0.9265**   | 0.9004           | 0.8958            |
| AUC           | —            | 0.9446           | **0.9590**        |
| 训练总耗时(s) | ~1871        | ~2001            | **~1438**         |

CrossEncoder 将句对拼接后联合建模，捕获跨句交互信息，F1 比最优 BiEncoder 高 +0.0261；BiEncoder 优势在于推理时仅需离线编码，检索场景效率更高。TripletLoss AUC（0.9590）高于 CosineEmbeddingLoss（0.9446），说明 Triplet 训练的向量空间排序能力更强，且训练耗时最短。

### 6.2 有监督模型 vs LLM 方法

| 对比维度     | CrossEncoder | BiEncoder Cosine | qwen3.7-plus Zero-shot | Qwen2.5-0.5B SFT |
| ------------ | ------------ | ---------------- | ---------------------- | ---------------- |
| Accuracy     | **0.9265**   | 0.9005           | 0.73                   | 0.855            |
| F1           | **0.9265**   | 0.9004           | 0.3721                 | 0.8547           |
| 是否调用 API | 否           | 否               | 是                     | 否               |

三种有监督模型的准确率（0.8958~0.9265）显著高于 LLM Zero-shot（0.73），差距约 +0.16~+0.20；LoRA SFT（0.855）经过领域微调后已接近有监督方法。BQ Corpus 为银行领域问句匹配，有监督训练能有效学习领域内语义相似分布；LLM Zero-shot 受制于领域特化能力，Recall 仅 0.24，漏匹配问题严重。

### 6.3 qwen3.7-plus Zero-shot vs Qwen2.5-0.5B LoRA SFT

| 方法                   | Accuracy  | F1(pos)    | Recall(pos) | Precision(pos) |
| ---------------------- | --------- | ---------- | ----------- | -------------- |
| qwen3.7-plus Zero-shot | 0.73      | 0.3721     | 0.2424      | **0.8000**     |
| Qwen2.5-0.5B LoRA SFT  | **0.855** | **0.8722** | —           | —              |

经过 LoRA 微调后，Qwen2.5-0.5B 准确率（0.855）和 F1（0.8722）均显著高于 qwen3.7-plus 零样本（0.73 / 0.3721），说明 BQ Corpus 银行领域充分的域内训练数据能有效弥补小模型的容量劣势。qwen3.7-plus 零样本精确率高（0.80）但召回率极低（0.24），整体偏向保守预测策略。

---

## 7. 结论

1. **CrossEncoder 整体最优**：验证集 F1=0.9265，联合建模句对交互是文本匹配精度最高的方案，适合对准确率要求高、实时性要求低的场景。
2. **BiEncoder TripletLoss 排序能力最强**：AUC=0.9590 为所有方法最高，训练耗时最短（~1438s），适合相似度排序或向量检索场景。
3. **有监督训练优于 LLM Zero-shot**：三种有监督方法 F1 均在 0.8958 以上，LLM Zero-shot 仅 0.3721（f1_pos）；经过 LoRA 微调的 Qwen2.5-0.5B（F1=0.8722）已接近 BiEncoder Cosine（0.9004），在 BQ Corpus 银行领域任务上，充足的领域数据 SFT 能大幅弥补小模型的能力劣势。
4. **小模型 SFT 反超大模型 Zero-shot**：经过 LoRA 微调的 Qwen2.5-0.5B 准确率（0.855）显著高于 qwen3.7-plus Zero-shot（0.73），说明在银行领域特化任务上，充分的域内训练数据能有效弥补参数量不足，SFT 增益超过底座规模效应。
5. **LLM Zero-shot 的核心瓶颈是 Recall**：qwen3.7-plus Recall(正)=0.2424，漏匹配问题严重；监督微调是提升召回的有效手段，Qwen2.5-0.5B LoRA SFT 后 F1(pos) 从 0.3721 提升至 0.8722，提升幅度显著。

