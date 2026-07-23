## 文档：`LCQWC` vs `BQ` 训练与评估对比报告

### 1. 概览

本对比基于 lcqwc 与 bq 中的训练/评估日志，覆盖以下内容：

- BiEncoder + CosineEmbeddingLoss
- BiEncoder + TripletLoss
- CrossEncoder + CrossEntropyLoss
- LLM zero-shot（llm_compare_results.json）

### 2. 数据集说明

- `LCQWC`
  - 口语化、通用问句匹配
  - 语言风格更随意、语义变化更大
  - 训练后模型判定阈值较高

- `BQ`
  - 银行金融领域问句匹配
  - 术语较多、场景更专业
  - 训练后模型判定阈值较低

### 3. Bert训练模型对比

#### 3.1 BiEncoder + CosineEmbeddingLoss

| 数据集 | epoch1 | epoch2 | epoch3 | 最终 val_f1 | 最优阈值 |
|------|-------|-------|-------|------------|--------|
| LCQWC | 0.7601 | 0.7842 | 0.7916 | 0.7916 | 0.79 |
| BQ | 0.8229 | 0.8514 | 0.8589 | 0.8589 | 0.67 |

- 结论：`BQ` 上效果显著优于 `LCQWC`
- 说明：`BQ` 任务更适合当前 BiEncoder cosine 训练配置

#### 3.2 BiEncoder + TripletLoss

| 数据集 | epoch1 | epoch2 | epoch3 | 最终 val_f1 | 最优阈值 |
|------|-------|-------|-------|------------|--------|
| LCQWC | 0.7601 | 0.8184 | 0.8184 | 0.8184 | 0.76 |
| BQ | 0.8514 | 0.8771 | 0.8625 | 0.8625 | 0.56 |

- 结论：Triplet 在两个数据集上都表现良好
- 细节：`BQ` 的 peak F1 更高，说明 `BQ` 负样本/正样本结构更适合当前 Triplet 构建方式

#### 3.3 CrossEncoder + CrossEntropyLoss

| 数据集 | epoch1 | epoch2 | epoch3 | 最终 val_f1 |
|------|-------|-------|-------|------------|
| LCQWC | 0.8162 | 0.8471 | 0.8543 | 0.8543 |
| BQ | 0.8389 | 0.8771 | 0.8849 | 0.8849 |

- 结论：CrossEncoder 是两者中表现最强的模型
- 说明：`BQ` 上 CrossEncoder 的最终表现最高，且收敛更快

#### 3.4 集内模型效果对比
| 数据集 | BiEncoder (Cosine) | BiEncoder (Triplet) | CrossEncoder (CrossEntropy) |
|------|--------------------|---------------------|----------------------------|
| LCQWC | 0.7916 | 0.8184 | 0.8543 |
| BQ | 0.8589 | 0.8625 | 0.8849 |   

- 结论：BiEncoder 使用 TripletLoss 表现更好; CrossEncoder 在三个模型中表现最佳


### 4. LLM 对比
#### 4.1 zero-shot 对比

| 数据集 | Accuracy | Precision_pos | Recall_pos | F1_pos |
|------|----------|---------------|------------|--------|
| LCQWC | 0.93 | 0.906 | 0.879 | 0.892 |
| BQ | 0.73 | 0.75 | 0.273 | 0.40 |

- 结论：LLM zero-shot 在 `LCQWC` 上远优于 `BQ`
- 说明：`BQ` 对当前 prompt/模型更不友好，尤其是正例召回严重不足

#### 4.2 SFT 对比

| 数据集 | Accuracy | F1_weighted | F1_pos |
|------|----------|---------------|-------|
| LCQWC | 0.855 | 0.855 | 0.853 |
| BQ | 0.78 | 0.78 | 0.79 |

- 结论：同样5000条数据训练，LLM SFT 在 `LCQWC` 上远优于 `BQ`
- 说明：`BQ` 对当前 prompt/模型更不友好


### 5. 关键发现

1. `BQ` Bert训练模型整体效果更好
   - BiEncoder/CrossEncoder 均比 `LCQWC` 高
   - 最优模型 `BQ CrossEncoder` 达到 `0.8849`

2. `LCQWC` Bert训练模型效果略低
   - 说明该数据集对当前模型/训练设置更难
   - 训练判定阈值较高（`0.76~0.79`），模型需要更严格的相似度判断

3. `LLM zero-shot` 的行为差异巨大
   - `LCQWC`：几乎接近理想状态
   - `BQ`：正例 F1 只有 `0.40`
   - 这表明两组数据集的语言风格和难点不同

4. `BQ` 的专业性更强
   - 从结果看，专业术语与金融场景更容易让 zero-shot LLM 预测保守
   - 但针对判别式 BERT 模型，`BQ` 反而更容易拟合

---

## 6. 结论摘要

- `BQ` 的 Bert训练模型表现优于 `LCQWC`
- `LCQWC` 的 LLM 表现显著优于 `BQ`
- 这说明两类数据集虽然都做文本匹配，但“任务风格”和“模型适配性”不同
- `LCQWC` 数据集总量23w+, `BQ` 数据集总量6w+, 但是`BQ`训练效果更好
- 这表明数据质量和任务适配性比单纯的数据量更重要
- 训练后模型判定阈值：`LCQWC` 较高（0.76~0.79），`BQ` 较低（0.56~0.67），说明 `LCQWC` 需要更严格的相似度判断


