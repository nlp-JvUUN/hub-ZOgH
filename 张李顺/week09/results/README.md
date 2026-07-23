# 实验结果

运行 `python evaluate.py` 后，此处生成：

- `comparison.csv`：三种检索方案的 `Recall@4`、`MRR@4` 与平均检索耗时；
- `query_details.csv`：10 个问题的人工标注相关页、实际检索页、逐题得分；
- `comparison.png`：检索质量与耗时的可视化对比图。

这些生成文件已被 `.gitignore` 排除，避免把实验产物推送到仓库。
