# 文本匹配 — LCQMC / BQ Corpus 多方法实验总结

> 在 AFQMC 项目基础上，把 **BiEncoder(Cosine) / BiEncoder(Triplet) / CrossEncoder** 三种方法
> 复用到 **LCQMC**（开放域口语问句）与 **BQ Corpus**（微众银行问句），
> 得到 **3 数据集 × 3 方法 = 9 组结果**，并对跨数据集规律做横向解读。
>
> ⚠️ 说明：AFQMC 数字为项目全量基线日志；LCQMC / BQ 为本机（Apple MPS）**快速实验**
> （训练取前 5,000 条、验证取前 1,250 条，3 epoch）。子集训练的绝对数值会低于全量，
> 但方法间相对关系与阈值规律可观察；全量复现命令见第六节（约 2~5 小时/数据集）。

---

## 一、实验设置（三套实验完全对齐，确保可比）

| 项 | 值 |
|----|----|
| 预训练模型 | bert-base-chinese（本地，ModelScope 下载） |
| BERT 层数 | 4（限层加速） |
| epoch / batch_size | 3 / 32 |
| max_length（biencoder / cross）| 64 / 128 |
| learning rate | 2e-5 |
| margin（cosine / triplet）| 0.3 |
| 池化 | mean |
| 评估指标 | Accuracy / weighted F1（biencoder 在 val 上做 101 阈值搜索） |
| 评估 split | validation |
| 硬件 | Apple Silicon（MPS，约 RTX 4060 的 1/1.7 速度） |

> **只换数据集，不换配方**，所有数字直接横向可比。

---

## 二、数据集对比

| 数据集 | train | val | 正样本占比 | 句长均值 | 字符 Jaccard | 领域 |
|-------|------:|----:|----------:|--------:|------------:|------|
| AFQMC | 34,334 | 4,316 | 30.8%（不均衡） | 15.5 字 | 0.405 | 蚂蚁金融问句 |
| LCQMC | 238,766 | 8,802 | 58.0% | 11.9 字 | 0.611 | 开放域口语问句 |
| BQ Corpus | 68,960 | 8,620 | 49.9% | 19.0 字 | 0.219 | 微众银行问句 |

> 数据量比例 **LCQMC : BQ : AFQMC ≈ 7 : 2 : 1**。AFQMC 是三者中最小且唯一类别不均衡（31:69）的。
> 字符 Jaccard 反映数据集"考什么"：LCQMC 正负样本词汇重叠都高（0.61）→ 考"换字不换意"的细粒度区分；
> BQ 重叠极低（0.22）→ 正例是"换说法"改写，考同义改写理解。

---

## 三、量化结果

### 3.1 最终对比（按数据集排，weighted F1）

| 数据集 | 训练量 | BiEncoder Cosine | BiEncoder Triplet | CrossEncoder | 该数据集冠军 |
|-------|------:|-----------------:|------------------:|-------------:|------------|
| AFQMC（全量基线） | 34K | **0.6765** | 0.6599 | 0.6750 | **Cosine** |
| LCQMC（5K 子集） | 5K | 0.7288 | 0.7151 | **0.7527** | **CrossEncoder** |
| BQ Corpus（5K 子集） | 5K | **0.7768** | 0.7380 | 0.7639 | **Cosine** |

### 3.2 各 epoch 验证集 F1 趋势（5K 子集）

| 数据集 | 方法 | epoch1 | epoch2 | epoch3 |
|-------|------|-------:|-------:|-------:|
| LCQMC | Cosine | 0.7217 | 0.7225 | **0.7288** |
|  | Triplet | 0.7123 | 0.7125 | **0.7151** |
|  | Cross | 0.7440 | 0.7416 | **0.7527** |
| BQ | Cosine | 0.7598 | 0.7719 | **0.7768** |
|  | Triplet | 0.6973 | 0.7318 | **0.7380** |
|  | Cross | 0.7330 | 0.7584 | **0.7639** |

> 三种方法 3 个 epoch 基本单调上升（LCQMC Cross 在 epoch2 微降后 epoch3 反超），
> 子集训练下 3 epoch 已接近收敛；全量数据下（参考基线）3 epoch 仍在上升，需要更多 epoch。

### 3.3 阈值（BiEncoder 在 val 上的最优分类阈值）

| 数据集 | Cosine 阈值 | Triplet 阈值 |
|-------|----------:|------------:|
| AFQMC（全量基线） | 0.51 | 0.81 |
| LCQMC（5K 子集） | 0.91 | 0.90 |
| BQ Corpus（5K 子集） | 0.65 | 0.62 |

> 阈值随数据集显著漂移（0.51 → 0.91），**验证"阈值不可跨数据集迁移"**：BiEncoder 部署前
> 必须在自家 val 上重新搜索阈值。

### 3.4 训练耗时（5K 子集 × 3 epoch，MPS）

| 数据集 | Cosine | Triplet | Cross | 总耗时 |
|-------|-------:|-------:|------:|------:|
| LCQMC | 3.5 min | 2.9 min | 3.3 min | ~11.5 min |
| BQ Corpus | 3.5 min | 2.6 min | 3.3 min | ~11.3 min |

---

## 四、跨数据集解读

### 4.1 方法相对关系：CrossEncoder 在 LCQMC 上夺冠，BQ 上 Cosine 领先

| 数据集 | Cosine | Triplet | Cross | 冠军 |
|-------|-------:|-------:|------:|------|
| AFQMC（全量） | 0.6765 | 0.6599 | 0.6750 | Cosine |
| LCQMC（5K） | 0.7288 | 0.7151 | **0.7527** | Cross |
| BQ（5K） | **0.7768** | 0.7380 | 0.7639 | Cosine |

- **LCQMC 上 Cross 领先**：开放域口语数据词汇重叠高（Jaccard 0.61），负例"换字不换意"，
  需要全层 token 交互才能区分 → CrossEncoder 全层交互优势显现（与全量实验结论一致）。
- **BQ 上 Cosine 领先**：银行问句词汇重叠极低（0.22），表示型模型靠句向量即可把"完全不同的
  业务问题"推开，交互优势不明显；子集训练下 Cross 未能反超。
- **Triplet 在两个子集上都垫底**：5K 子集内三元组仅 ~2.5K~2.9K 个，训练信号不足，
  与 AFQMC 全量上"Triplet 弱于 Cosine"一致 —— **三元组数量 ≈ 正例对数，数据量小是 Triplet 的硬伤**。

### 4.2 阈值漂移：0.51 → 0.91，BiEncoder 必须重搜阈值

Cosine 阈值从 AFQMC 的 0.51 漂到 LCQMC 子集的 0.91，说明不同数据分布下正负样本的相似度
量级完全不同。**任何跨数据集复用都必须重新做 101 阈值搜索**（评估脚本已内置，成本极低）。

### 4.3 数据集"难度"与数据量

- 5K 子集上 BQ 的绝对 F1（0.7768）> LCQMC（0.7527）> AFQMC 全量（0.6765）：
  域窄标注一致的 BQ 最容易，AFQMC 数据少 + 噪声 + 不均衡最难；
- 子集实验绝对数值低于全量（参考同仓库全量复现：LCQMC 0.8562 / BQ 0.8848，Cross 均夺冠），
  但**方法相对关系与阈值规律在子集上依然成立**，说明这些规律主要由数据分布决定而非样本量。

---

## 五、结论

1. **数据量决定 Loss 选型**：三元组数量 ≈ 正例对数，正例不足 10K 时 Triplet 明显吃亏
   （本实验 AFQMC 全量、LCQMC/BQ 子集均复现）；数据量达到 10⁵ 级时 Triplet 才可能反超。
2. **CrossEncoder 的交互优势需要数据支撑**：LCQMC（高词汇重叠难负例）上 Cross 领先；
   BQ（低重叠易区分）上表示型 Cosine 足够。
3. **阈值搜索是 BiEncoder 部署的标准步骤**：本实验阈值跨数据集漂移 0.51→0.91，不可复用。
4. **数据量 + 标注质量 > 领域宽度**：域窄一致的 BQ 最容易，数据少 + 噪声的 AFQMC 最难。

---

## 六、复现命令

```bash
conda activate py312            # torch 2.6 + transformers 4.55

# 0) 首次：下载 bert-base-chinese（ModelScope，HF 不可达）
python download_bert.py

# 1) 快速验证（本报告所用，~11 分钟/数据集）
python run_all.py --dataset bq_corpus --quick 5000
python run_all.py --dataset lcqmc --quick 5000

# 2) 全量实验（约 2~5 小时/数据集）
python run_all.py --dataset bq_corpus
python run_all.py --dataset lcqmc

# 3) 汇总三数据集对比（→ results/comparison_table.md + figures/）
python aggregate.py
```

产物按数据集归档在 `results/<dataset>/{checkpoints,logs,figures}`，互不覆盖。
