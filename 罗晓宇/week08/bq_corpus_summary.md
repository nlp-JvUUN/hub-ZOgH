# 模型训练与评估分析报告

## 1. 目录结构与结果文件

- `outputs/checkpoints/`
  - `biencoder_cosine_best.pt`
  - `biencoder_triplet_best.pt`
  - `crossencoder_best.pt`
- `outputs/logs/`
  - `biencoder_cosine_log.json`
  - `biencoder_triplet_log.json`
  - `crossencoder_log.json`
  - `method_comparison.json`
  - `llm_compare_results.json`
  - `sft_results.json`
  - `train_sft.json`
- `outputs/sft_adapter/`
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `chat_template.jinja`
  - `tokenizer.json`
  - `tokenizer_config.json`

## 2. 训练结果概览

### 2.1 CrossEncoder
- 最终验证准确率：`0.8849`
- 最终验证 F1：`0.8849`
- 训练损失下降：`0.4977` → `0.2538`
- 训练准确率上升：`0.7494` → `0.8966`
- 每 epoch 约耗时：`363s`

### 2.2 BiEncoder (TripletLoss)
- 最终验证准确率：`0.8625`
- 最终验证 F1：`0.8625`
- 训练损失下降：`0.1244` → `0.0339`
- 最优阈值：`0.56`
- 每 epoch 约耗时：`308s`

### 2.3 BiEncoder (CosineEmbeddingLoss)
- 最终验证准确率：`0.8592`
- 最终验证 F1：`0.8589`
- 训练损失下降：`0.2316` → `0.1522`
- 最优阈值：`0.67`
- 每 epoch 约耗时：`430s`

## 3. 方法对比结果

来自 `outputs/logs/method_comparison.json` 的最终对比：

| 方法 | Accuracy | F1 | AUC / 其他 |
|---|---|---|---|
| BiEncoder (CosineEmbeddingLoss) | 0.8592 | 0.8589 | AUC = 0.9260 |
| BiEncoder (TripletLoss) | 0.8625 | 0.8625 | AUC = 0.9295 |
| CrossEncoder (CrossEntropyLoss) | 0.8849 | 0.8849 | - |

## 4. LLM 与 SFT 对比结果

### 4.1 LLM 对比结果
来自 `outputs/logs/llm_compare_results.json`：
- 模型：`deepseek-v4-flash`
- 样本数：`100`
- Accuracy：`0.73`
- Positive precision：`0.75`
- Positive recall：`0.2727`
- Positive F1：`0.40`

> 结论：该 LLM 在正类召回上表现较弱，尽管整体准确率尚可，但正例识别存在明显不足。

### 4.2 SFT 结果
来自 `outputs/logs/sft_results.json`：
- Accuracy：`0.78`
- F1(weighted)：`0.7808`
- Positive F1：`0.7905`
- 样本数：`200`
- parse_fail：`0`

## 5. 核心结论

1. **CrossEncoder 最优**：在当前验证指标上，CrossEncoder 的准确率和 F1 均领先，是最强的单模型方案。
2. **BiEncoder Triplet 性能优于 Cosine**：TripletLoss 的 BiEncoder 在 F1 和 AUC 上略胜 CosineEmbeddingLoss，推荐作为双塔检索的首选。
3. **LLM 目前不是最优评分方案**：尽管 LLM 对比结果准确率可观，但正类召回率低，说明它更适合用于候选生成而非精确相似判断。
4. **SFT 已有一定效果**：现有 SFT 模型在 200 条样本上的 accuracy 和 F1 领先于 LLM zero-shot，但仍低于当前 CrossEncoder 的最高值。

## 6. 推荐策略

### 6.1 生产部署建议

- 若追求最高质量：直接使用 `CrossEncoder` 做判定。
- 若需要兼顾效率与准确率：建议使用 `BiEncoder (TripletLoss)` 做检索候选，再用 `CrossEncoder` 对 top-K 结果做重排序。
- 若希望降低在线延迟：部署 `BiEncoder` 向量检索并使用阈值 `0.56`。

### 6.2 后续优化方向

- 进一步训练 CrossEncoder 多轮 epoch，观察 val 指标是否还能继续提升。
- 对 `BiEncoder` 进行更长训练，比较 Triplet 与 Cosine 的长期收敛差异。
- 进一步收集和分析 LLM 与 SFT 的错误样本，尤其是正类漏判与负类误判。
- 如果允许，可结合多模型投票或级联方案：BiEncoder 先检索 + CrossEncoder 复判。

## 7. 参考文件

- `outputs/logs/biencoder_cosine_log.json`
- `outputs/logs/biencoder_triplet_log.json`
- `outputs/logs/crossencoder_log.json`
- `outputs/logs/method_comparison.json`
- `outputs/logs/llm_compare_results.json`
- `outputs/logs/sft_results.json`

---

*生成时间：2026-06-20*