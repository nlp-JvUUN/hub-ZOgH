# LCQMC 与 BQ Corpus 句对匹配技术报告

## 1. 任务范围

任务是 pair classification：输入两个中文句子，判断语义是否匹配。所有方法统一输出连续匹配分数，再在 validation 上选择最大 F1 阈值，在完整 test 上评估。项目不构造全库相关性标注，也不报告检索排序指标。

## 2. 数据

| 数据 | split | 原始数量 | 正例率 | 句1/句2 P95长度 |
|---|---|---:|---:|---:|
| LCQMC | train | 238,766 | 0.580 | 18/20 |
| LCQMC | validation | 8,802 | 0.500 | 19/19 |
| LCQMC | test | 12,500 | 0.500 | 14/15 |
| BQ | train | 68,960 | 0.499 | 24/26 |
| BQ | validation | 8,620 | 0.502 | 24/26 |
| BQ | test | 8,620 | 0.508 | 25/26 |

BQ 中存在少量包含嵌入换行或制表符的拼接坏记录。评估确定性剔除这些记录，因此 BQ 实际使用 validation 8,619 条、test 8,618 条；LCQMC 使用完整 8,802/12,500 条。原始文件不修改。

## 3. 模型

- BiEncoder Cosine：`BAAI/bge-small-zh-v1.5`，`CosineEmbeddingLoss`。
- BiEncoder Triplet：相同底座，余弦距离 Triplet Loss。
- CrossEncoder：相同 BGE backbone 拼接句对，增加二分类头。
- BiEncoder+BM25：字符 unigram+bigram BM25 与最佳 BiEncoder 分数按 `0.7/0.3` 融合。
- LLM Zero-shot：`Qwen2-0.5B-Instruct` 输出“是/否”的 logit 差。
- LLM SFT LoRA：只训练 `q_proj/v_proj`，rank 8、alpha 16、200 steps。

| 数据 | Bi-Cosine | Bi-Triplet | CrossEncoder | Qwen LoRA |
|---|---:|---:|---:|---:|
| LCQMC | 8.06s | 10.01s | 40.12s | 41.91s |
| BQ | 8.78s | 8.97s | 39.88s | 41.08s |

所有单模型训练均低于 42 秒，并设置 240 秒硬上限。

## 4. 完整测试集结果

### LCQMC

| 方法 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Bi-Cosine | 0.778 | 0.712 | 0.933 | 0.808 | 0.914 | 0.922 |
| Bi-Triplet | 0.827 | 0.781 | 0.908 | 0.840 | 0.924 | 0.927 |
| CrossEncoder | 0.791 | 0.719 | **0.956** | 0.821 | **0.928** | **0.931** |
| BiEncoder+BM25 | **0.836** | **0.814** | 0.870 | **0.841** | 0.913 | 0.907 |
| LLM Zero-shot | 0.766 | 0.709 | 0.903 | 0.794 | 0.877 | 0.875 |
| LLM SFT LoRA | 0.780 | 0.717 | 0.925 | 0.808 | 0.902 | 0.905 |

LCQMC 若以阈值后的 F1 为目标，BiEncoder+BM25 最佳；CrossEncoder 的 ROC-AUC/PR-AUC 最佳。

### BQ Corpus

| 方法 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Bi-Cosine | 0.763 | 0.716 | **0.883** | 0.791 | 0.858 | 0.850 |
| Bi-Triplet | **0.787** | 0.750 | 0.872 | **0.806** | **0.880** | **0.878** |
| CrossEncoder | 0.785 | 0.747 | 0.872 | 0.805 | 0.876 | 0.877 |
| BiEncoder+BM25 | 0.783 | **0.758** | 0.840 | 0.797 | 0.870 | 0.863 |
| LLM Zero-shot | 0.633 | 0.598 | 0.847 | 0.701 | 0.733 | 0.749 |
| LLM SFT LoRA | 0.734 | 0.703 | 0.825 | 0.759 | 0.822 | 0.827 |

BQ 上使用同 anchor 显式负例训练的 Bi-Triplet 在 F1、ROC-AUC 和 PR-AUC 上略优于 CrossEncoder。LoRA 相比 Zero-shot 明显改善，但仍未超过小型判别模型。

## 5. Badcase

对完整测试集的全部预测错误区分 false positive 和 false negative，保存模型分数、分类阈值以及错误预测距离阈值的大小。项目不自动判断错误原因，原因分析由人工阅读具体句对完成。完整结果位于 `outputs/<dataset>/badcases.csv`、`badcase_summary.csv` 与 `badcase_analysis.md`。

## 6. 运行

```powershell
conda activate ai_learning_1
python pipeline.py all
```

结果以 `outputs/<dataset>/pair_metrics.csv` 为准；分类效果图为 `outputs/figures/<dataset>_pair_comparison.png`。
