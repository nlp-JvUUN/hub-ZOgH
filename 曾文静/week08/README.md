# Week08 — 文本匹配：LCQMC / BQ Corpus 多方法实验

> 在 AFQMC 项目基础上，把 **BiEncoder(Cosine) / BiEncoder(Triplet) / CrossEncoder** 三种方法
> 原样复用到 **LCQMC** 和 **BQ Corpus**，与已跑好的 AFQMC 基线形成 **3 数据集 × 3 方法 = 9 组**横向对比。

## 一、目录结构

```
week08/
├── download_bert.py     # 下载 bert-base-chinese（ModelScope 镜像，HF 被墙）
├── run_all.py           # 一键训练+评估一个数据集（产物自动归档，互不覆盖）
├── aggregate.py         # 汇总 9 组日志 → 对比表 + 对比图
├── data/                # 拷贝项目（无需重新下载）
│   ├── lcqmc/           # 238,766 训练对（开放域口语问句）
│   └── bq_corpus/       #  68,960 训练对（微众银行问句）
├── src/                 # 拷贝项目（唯一改动：设备检测支持 Apple MPS）
├── pretrain_models/     # bert-base-chinese（首次运行 download_bert.py 下载）
└── results/
    ├── afqmc/           # 已跑好的基线（日志，不重训）
    ├── lcqmc/  bq_corpus/   # 本作业产物：checkpoints / logs / figures
    ├── comparison_table.md  # aggregate.py 生成
    └── figures/              # f1_comparison.png / f1_epochs.png
```

## 二、复现步骤

```bash
conda activate py312          # 需要 torch 2.6+（本机在 /opt/miniconda3/envs/py312）

# 0) 首次：下载预训练模型（约 400MB，ModelScope）
python download_bert.py

# 1) 训练 + 评估（每个数据集约 3 个命令，或直接全量跑）
python run_all.py --dataset bq_corpus        # BQ，约 40~90 分钟（MPS）
python run_all.py --dataset lcqmc            # LCQMC，最久（建议后台跑）
python run_all.py --dataset lcqmc --quick 5000   # 快速验证流程（5~10 分钟）

# 2) 汇总三数据集对比
python aggregate.py          # → results/comparison_table.md + figures/
```

## 三、实验设置（与 AFQMC 基线完全对齐）

| 项 | 值 |
|----|----|
| 预训练模型 | bert-base-chinese（本地） |
| BERT 层数 | 4（`--layers 12` 可换全量） |
| epoch / batch | 3 / 32 |
| max_length | BiEncoder 64 / CrossEncoder 128 |
| lr | 2e-5 |
| margin | 0.3 |
| 池化 | mean |
| 评估 | validation 集，Accuracy + weighted F1，BiEncoder 101 阈值搜索 |

> 原则：**只换数据集，不换配方**，9 组数字直接横向可比。


## 四、结果

**9 组结果（weighted F1）**：AFQMC 为全量基线；LCQMC / BQ 为本机快速实验（5K 训练子集 × 3 epoch）。

| 数据集 | BiEncoder Cosine | BiEncoder Triplet | CrossEncoder | 冠军 |
|-------|-----------------:|------------------:|-------------:|------|
| AFQMC（全量基线） | **0.6765** | 0.6599 | 0.6750 | Cosine |
| LCQMC（5K 子集） | 0.7288 | 0.7151 | **0.7527** | Cross |
| BQ Corpus（5K 子集） | **0.7768** | 0.7380 | 0.7639 | Cosine |

- 完整对比表与图表：`results/comparison_table.md` + `results/figures/`（`aggregate.py` 自动生成）
- 实验报告：`SUMMARY.md`（含各 epoch 趋势、阈值、跨数据集解读）
- 全量复现：`python run_all.py --dataset bq_corpus` / `--dataset lcqmc`（约 2~5 小时/数据集）
