# week07 — 中文序列标注(NER)模型训练

在 **cluener2020** 数据集上实现序列标注模型训练，对比 **BERT+Linear（基线）** 与 **BERT+CRF** 两种解码方式的效果差异。

## 目录结构

```
week07/
├── download_data.py    # 下载 cluener2020 并解析为统一格式
├── dataset.py          # span标注→BIO标签 + BERT子词对齐
├── model.py            # BERT + Linear / BERT+CRF 两种解码头
├── metrics.py          # seqeval entity-level F1、非法BIO统计、实体还原
├── train.py            # 训练脚本(--use_crf 切换两种模型)
├── evaluate.py         # 评估 + 对比 + 推理demo
├── requirements.txt    # 依赖
└── data/               # 下载的数据(已加入.gitignore, 不推送到git)
```

## 环境准备

```bash
pip install -r requirements.txt
# 国内下载 bert-base-chinese 慢时, 设置镜像:
#   export HF_ENDPOINT=https://hf-mirror.com
```

## 使用步骤

### 1. 下载数据

```bash
python download_data.py
# 生成 data/cluener/{train,validation,test}.json (约1.1万/1343/1345条)
```

### 2. 训练

```bash
# BERT + Linear 基线
python train.py

# BERT + CRF
python train.py --use_crf

# 常用参数: --epochs 5 --lr 3e-5 --batch_size 16 --max_length 128
# 快速实验: --num_train 2000 (只用前2000条, 先跑通流程)
```

输出：
- `outputs/checkpoints/best_linear.pt` / `best_crf.pt`
- `outputs/figures/train_linear.png` / `train_crf.png`（损失与验证F1曲线）
- `outputs/logs/train_*.json`

### 3. 评估与对比

```bash
python evaluate.py             # 评估 BERT+Linear
python evaluate.py --use_crf   # 评估 BERT+CRF
python evaluate.py --compare   # 两模型对比表 + 逐类型F1 + 非法序列统计
```

评估输出：entity-level P/R/F1、逐实体类型 F1、非法 BIO 序列统计、推理 Demo。

## 实验要点（写报告/对比时用）

1. **评估口径**：使用 seqeval 的 **entity-level F1**（按完整实体边界匹配），不是 token 级准确率；
2. **CRF 的价值**：Linear 头会有约 1~2% 的序列产生非法 BIO 转移（`I-X` 开头、跨类型转移），CRF 的 Viterbi 解码保证为 **0**；
3. **逐类型分析**：不同实体类型 F1 差异大（如 name 通常最好、address/company 较难），可结合错误案例分析；
4. **标签对齐**：BERT 子词切分后，非首子词在 Linear 模式用 `-100` 屏蔽 loss；CRF 模式沿用首子词标签。

## 常见问题

- `ModuleNotFoundError: seqeval / torchcrf` → `pip install seqeval pytorch-crf`
- bert-base-chinese 下载失败 → 设置 `HF_ENDPOINT=https://hf-mirror.com`
- 显存不足(OOM) → 调小 `--batch_size`(16→8) 或 `--max_length`(128→64)，cluener2020 的 P95 文本长度约 60 字
- `seqeval` 报前缀错误 → 标签必须是 `O`/`B-type`/`I-type`（连字符，不能是下划线）
