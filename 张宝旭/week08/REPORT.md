# Week 8 文本匹配作业报告

## 一、实验目标

在 **BQ_CORPUS**（银行金融领域中文语义匹配数据集）上对比三种文本匹配方法：

1. BiEncoder + CosineEmbeddingLoss
2. BiEncoder + TripletLoss
3. CrossEncoder + CrossEntropyLoss

---

## 二、实验配置（控制变量）

| 配置项 | 取值 |
|--------|------|
| 预训练模型 | bert-base-chinese |
| BERT 层数 | 4 层（限层加速）|
| 训练轮数 | 3 epochs |
| Batch Size | 32 |
| 学习率 | 2e-5（BERT 层）|
| 优化器 | AdamW + Linear Warmup |
| 评估集 | BQ_CORPUS validation |
| 设备 | CPU |

---

## 三、训练过程（每 epoch 指标）

### 3.1 BiEncoder + CosineEmbeddingLoss

| Epoch | train_loss | val_acc | val_f1 | threshold | 用时 |
|-------|-----------|---------|--------|-----------|------|
| 1 | 0.2323 | 0.8225 | 0.8225 | 0.72 | 56 min |
| 2 | 0.1719 | 0.8559 | 0.8556 | 0.64 | 57 min |
| **3** | **0.1522** | **0.8633** | **0.8631** | **0.66** | 57 min |

### 3.2 BiEncoder + TripletLoss

| Epoch | train_loss | val_acc | val_f1 | threshold | 用时 |
|-------|-----------|---------|--------|-----------|------|
| 1 | 0.1248 | 0.8219 | 0.8218 | 0.55 | 42 min |
| 2 | 0.0506 | 0.8513 | 0.8512 | 0.54 | 40 min |
| **3** | **0.0341** | **0.8575** | **0.8574** | **0.51** | 43 min |

### 3.3 CrossEncoder + CrossEntropyLoss

| Epoch | train_loss | train_acc | val_acc | val_f1 | 用时 |
|-------|-----------|-----------|---------|--------|------|
| 1 | 0.4967 | 0.7489 | 0.8319 | 0.8317 | 59 min |
| 2 | 0.3372 | 0.8534 | 0.8712 | 0.8712 | 57 min |
| **3** | **0.2571** | **0.8946** | **0.8811** | **0.8811** | 59 min |

---

## 四、最终对比结果（BQ_CORPUS validation）

| 方法 | Accuracy | F1 (weighted) | AUC | 决策方式 |
|------|----------|---------------|-----|---------|
| BiEncoder (CosineEmbeddingLoss) | 0.8633 | 0.8631 | **0.9262** | threshold=0.66 |
| BiEncoder (TripletLoss) | 0.8575 | 0.8574 | **0.9294** | threshold=0.51 |
| **CrossEncoder (CrossEntropyLoss)** | **0.8811** | **0.8811** | — | argmax |

对比图：
- `outputs/figures/method_comparison_bar.png` — 三方法 Accuracy/F1 对比柱状图
- `outputs/figures/biencoder_sim_distributions.png` — BiEncoder 正负样本相似度分布

---

## 五、结果分析

### 5.1 关键发现

1. **CrossEncoder 精度最高**（F1=0.8811），比 BiEncoder Cosine 高 1.8 个百分点
   - 原因：两句在 BERT 每一层都跨句交互（Self-Attention），表达能力更强

2. **BiEncoder 两种 Loss 差距很小**（F1 仅相差 0.6%）
   - Cosine 在 Accuracy/F1 略高（0.8631 vs 0.8574）
   - Triplet 在 AUC 略高（0.9294 vs 0.9262）→ 排序能力更强
   - **结论**：在 BQ_CORPUS 数据量下，两种 Loss 各有千秋

3. **训练 loss 差异显著**
   - Triplet 最终 train_loss=0.034，远低于 Cosine 的 0.152
   - Triplet loss 数值更小是因为 margin 机制下大部分三元组已满足约束 → 不代表泛化更好

### 5.2 阈值变化趋势

- Cosine 阈值收敛在 **0.66** 附近（高阈值 → 模型把正例相似度推得很高）
- Triplet 阈值收敛在 **0.51** 附近（低阈值 → 整体相似度分布偏低，但正负之间区分度仍很好）

### 5.3 速度 vs 精度权衡

| 方法 | 推理可向量化 | 适用场景 |
|------|------------|---------|
| BiEncoder | ✅ 可预计算句向量 | 大规模检索（RAG Recall）|
| CrossEncoder | ❌ 每对都要过一次 BERT | 精排（Reranker）|

**生产推荐架构**：BiEncoder 召回 Top-K → CrossEncoder 精排，兼顾速度和精度。

---

## 六、结论

1. **精度排序**：CrossEncoder > BiEncoder(Cosine) ≈ BiEncoder(Triplet)
2. **AUC 排序**：Triplet > Cosine（说明 Triplet 学到的相似度排序更准）
3. BQ_CORPUS 比 AFQMC 上指标显著更高（这里 F1≈0.88 vs AFQMC 仅 0.65）— 说明 BQ_CORPUS 类别分布更均衡、句子相似性更明显
4. 实际工业部署推荐 **BiEncoder 召回 + CrossEncoder 精排** 的两阶段方案

---

## 七、产出文件

```
outputs/
├── checkpoints/
│   ├── biencoder_cosine_best.pt      (174 MB)
│   ├── biencoder_triplet_best.pt     (174 MB)
│   └── crossencoder_best.pt          (174 MB)
├── logs/
│   ├── biencoder_cosine_log.json
│   ├── biencoder_triplet_log.json
│   ├── crossencoder_log.json
│   └── method_comparison.json
└── figures/
    ├── method_comparison_bar.png         ← 三方法对比柱状图
    └── biencoder_sim_distributions.png   ← BiEncoder 正负样本相似度分布
```

---

## 八、复现命令

```bash
# 训练三个模型
python src/train_biencoder.py --data_dir data/bq_corpus --loss cosine
python src/train_biencoder.py --data_dir data/bq_corpus --loss triplet
python src/train_crossencoder.py --data_dir data/bq_corpus

# 对比结果
python src/compare_methods.py --data_dir data/bq_corpus
```
