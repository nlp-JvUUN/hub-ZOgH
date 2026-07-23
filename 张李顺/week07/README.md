# LCQMC 与 BQ Corpus 句对匹配

项目只处理 pair classification：输入 `sentence1` 和 `sentence2`，输出是否匹配。包含数据分析、BiEncoder Cosine、BiEncoder Triplet、CrossEncoder、BiEncoder+BM25、LLM Zero-shot、Qwen LoRA SFT、完整测试集评估和 badcase。

## 环境

```powershell
conda activate ai_learning_1
```

代码优先复用本机缓存的 `BAAI/bge-small-zh-v1.5` 与 `Qwen/Qwen2-0.5B-Instruct`。

## 运行

```powershell
python pipeline.py all
```

分阶段运行：

```powershell
python pipeline.py analyze
python pipeline.py train
python pipeline.py evaluate
python pipeline.py report
```

训练有 240 秒硬停止条件。评估使用完整 validation 选择阈值，使用完整 test 计算 Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC 和混淆矩阵。完整 LLM 评估耗时不属于训练时间。

## 输出

- `outputs/data_summary.csv`：数据统计。
- `outputs/<dataset>/pair_metrics.csv`：完整分类指标。
- `outputs/<dataset>/pair_predictions.csv`：完整测试集预测。
- `outputs/<dataset>/badcases.csv`：完整 FP/FN 错误样本、分数、阈值和错误置信度。
- `outputs/<dataset>/badcase_summary.csv`：各模型 FP/FN 数量与错误率。
- `outputs/figures`：数据分析、分类效果、训练时间和 badcase 图。
- `docs/TRIPLET_SAMPLING.md`：Triplet 取样策略。
- `docs/TECHNICAL_REPORT.md`：技术报告。
