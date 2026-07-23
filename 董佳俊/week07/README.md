# 文本分类 —— 对比不同训练方法效果

## 任务
中文情感二分类（正面/负面），基于酒店评论数据。

## 对比方法（共 9 种，3 大类）

### 传统机器学习（特征工程 + 浅层分类器）
| 方法 | 说明 |
|------|------|
| TF-IDF + Logistic Regression | 线性基线 |
| TF-IDF + SVM | 最大间隔分类器 |
| TF-IDF + XGBoost | 梯度提升树 |

### BERT 微调（3 种训练策略）
| 方法 | 说明 |
|------|------|
| BERT 全量微调 (full) | 102M 参数全部更新 |
| BERT 冻结+分类头 (freeze) | 只训练分类头 ~0.6M |
| BERT LoRA 微调 (lora) | 低秩适配器 ~0.3M |

### LLM（大模型）
| 方法 | 说明 |
|------|------|
| LLM zero-shot | 不训练，纯 prompt |
| LLM few-shot | 不训练，4 个示例引导 |
| Qwen2.5 LoRA SFT | 指令微调 ~0.8M |

## 快速开始

```bash
# 1. 传统 ML baseline（不需要 GPU）
python src_traditional/ml_baseline.py

# 2. BERT 微调（三种策略各跑一次）
python src_bert/bert_trainer.py --method full --epochs 3
python src_bert/bert_trainer.py --method freeze --epochs 3
python src_bert/bert_trainer.py --method lora --epochs 3

# 3. LLM 提示（需要 DEEPSEEK_API_KEY 环境变量）
python src_llm/prompt_classify.py --n_samples 100

# 4. LLM 指令微调（可选，需要 GPU）
python src_llm/sft_finetune.py --num_train 500 --epochs 2

# 5. 汇总对比
python compare_all.py
```

## 目录结构

```
├── data_loader.py              # 统一数据加载（HuggingFace + fallback）
├── compare_all.py              # 汇总对比所有方法
│
├── src_traditional/
│   └── ml_baseline.py          # TF-IDF + LR/SVM/XGBoost
│
├── src_bert/
│   ├── bert_model.py           # BERT 分类模型（full/freeze/lora）
│   └── bert_trainer.py         # 训练 + 评估
│
├── src_llm/
│   ├── prompt_classify.py      # LLM zero-shot / few-shot
│   └── sft_finetune.py         # Qwen2.5 LoRA 指令微调
│
└── outputs/
    ├── checkpoints/            # 模型 checkpoint
    ├── logs/                   # 评估结果 JSON
    └── figures/                # 对比图表
```

## 依赖

```bash
pip install torch transformers tqdm scikit-learn openai
pip install peft xgboost matplotlib  # 可选
```
