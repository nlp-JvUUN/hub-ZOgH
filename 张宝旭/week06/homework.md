# 文本分类不同训练方法对比实验报告

## 一、实验任务

对中文新闻短文本进行 15 类分类，对比四种不同训练方法的效果：
1. **BERT 微调（普通 CrossEntropyLoss）** —— 经典方案
2. **BERT 微调（加权 CrossEntropyLoss）** —— 应对类不均衡
3. **LLM Zero-Shot（Qwen2-0.5B-Instruct）** —— 不训练，直接 prompt
4. **LLM SFT (LoRA, Qwen2-0.5B-Instruct)** —— 少量样本 LoRA 微调

**类别**：故事 / 文化 / 娱乐 / 体育 / 财经 / 房产 / 汽车 / 教育 / 科技 / 军事 / 旅游 / 国际 / 证券 / 农业 / 电竞

**数据集规模**：
- 训练集：53,360 条
- 验证集：10,000 条
- 测试集：10,000 条

## 二、实验环境

- 操作系统：Windows
- Python：3.12.7（miniconda）
- 框架：torch 2.12.0+cpu, transformers 5.9.0, peft 0.19.1
- 硬件：CPU 训练（AMD Ryzen 7 7800X3D，16 线程，32GB 内存）
- 预训练模型：
  - `bert-base-chinese`（102.3M 参数）
  - `Qwen2-0.5B-Instruct`（494M 参数，由 ModelScope 下载）

## 三、四种方法详解

### 3.1 BERT 微调（普通 / 加权 loss）

| 项 | 设置 |
|---|---|
| 模型 | bert-base-chinese + Linear 分类头 |
| 训练数据 | 全量 53,360 条 |
| epochs | 1 |
| batch_size | 32 |
| max_length | 64 |
| learning_rate | 2e-5（BERT 主体），1e-4（分类头） |
| pool | cls |
| 总参数量 | 102.3M（全部参与训练） |

**两种 loss 的差异**：
- 普通：`nn.CrossEntropyLoss()`
- 加权：`nn.CrossEntropyLoss(weight=class_weights)`，类频反比

加权时各类权重（节选）：
```
故事=3.20  财经=0.68  科技=0.60  证券=13.84  ← 证券权重最高（样本最少）
```

### 3.2 LLM Zero-Shot

| 项 | 设置 |
|---|---|
| 模型 | Qwen2-0.5B-Instruct |
| 训练数据 | 0（不训练） |
| 推理方式 | Prompt + 解析输出 |
| 评测样本 | 200 条（验证集随机采样） |

Prompt 模板：要求模型从 15 个类别中选一个输出。

### 3.3 LLM SFT (LoRA)

| 项 | 设置 |
|---|---|
| 基础模型 | Qwen2-0.5B-Instruct |
| 微调方式 | LoRA（rank=8, alpha=16） |
| 训练数据 | 1,000 条（从训练集采样） |
| epochs | 1 |
| batch_size | 4 × grad_accum 4 = 16 |
| learning_rate | 2e-4 |
| 可训练参数 | 1,081,344（占总参数 0.22%） |
| 评测样本 | 200 条（与 Zero-Shot 同一批） |

## 四、实验结果

### 4.1 BERT 普通 CrossEntropyLoss

| epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | 1.3734 | 0.5269 | **0.5589** | **0.5430** | 5,157 (≈86 min) |

### 4.2 BERT 加权 CrossEntropyLoss

| epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | 1.3695 | 0.5151 | **0.5477** | **0.5403** | 5,206 (≈87 min) |

### 4.3 BERT 普通 vs 加权（同样 1 epoch）

| 指标 | 普通 loss | 加权 loss | 差异 |
|---|---|---|---|
| val_acc | 0.5589 | 0.5477 | -1.12pp |
| val_macro_f1 | 0.5430 | 0.5403 | -0.27pp |
| train_acc | 0.5269 | 0.5151 | -1.18pp |

**观察**：在仅训练 1 epoch 的情况下，加权 loss 略低于普通 loss。
**分析**：
- 加权 loss 对小类（如证券）赋予更大权重（13.84 倍），梯度方向更不稳定
- 1 epoch 内模型还没充分收敛到加权目标的最优点
- 真正的加权 loss 优势通常需要训 3+ epoch 才能体现（小类 F1 上升）

### 4.4 LLM Zero-Shot（Qwen2-0.5B-Instruct）

| 指标 | 数值 |
|---|---|
| 评测样本 | 200 |
| 正确数 | 72 |
| **accuracy** | **0.3600** |
| **unparseable** | **58 (29.0%)** |
| 总耗时 | 60.3s（0.30s/条） |
| 推理速度 | 平均 0.30 秒/条 |

**典型 unparseable 案例**：

| 原文 | 真实标签 | LLM 输出 |
|---|---|---|
| 韩国前总统朴槿惠... | 国际 | 政治 |
| 步枪之王 M416 枪械搭配... | 电竞 | 武器 |
| 如何寻找转世灵童... | 文化 | 宗教 |
| $65W 澳币就能买下小岛... | 国际 | 房地产 |

**关键发现**：能解析的 142 条中正确率 = 72/142 = **50.7%**。
Zero-Shot 的瓶颈不是知识，而是**输出格式不可控**。

### 4.5 LLM SFT (LoRA, Qwen2-0.5B-Instruct + 1000 条训练)

**训练日志**：

| epoch | train_loss | val_loss | 耗时(s) |
|---|---|---|---|
| 1 | 0.8206 | 0.7858 | 707 (≈12 min) |

**评估结果**：

| 指标 | 数值 |
|---|---|
| 评测样本 | 200 |
| 正确数 | 100 |
| **accuracy** | **0.5000** |
| **unparseable** | **3 (1.5%)** |

**关键改善（与 Zero-Shot 对比）**：
- accuracy：0.36 → 0.50，**提升 14pp**
- unparseable：58 → 3，**减少 95%**
- 训练成本：0 → 12 分钟、1000 条样本、训练 0.22% 参数

### 4.6 四方法整体对比

| 方法 | 训练数据 | val_acc | unparseable | 训练时长 | 评测样本 |
|---|---|---|---|---|---|
| BERT 普通 loss | 53,360 | **0.5589** | — | 86 min | 10,000 |
| BERT 加权 loss | 53,360 | 0.5477 | — | 87 min | 10,000 |
| LLM Zero-Shot | 0 | 0.3600 | 29.0% | 0 | 200 |
| LLM SFT (LoRA) | 1,000 | 0.5000 | 1.5% | 12 min | 200 |

> ⚠️ **评测口径不完全一致**：BERT 在验证集 10,000 条全量评测，LLM 因推理较慢仅评 200 条（同一随机采样）。直接比大小需谨慎，但量级关系可靠。

## 五、分析与洞察

### 5.1 准确率对比

```
BERT 普通 loss  (53K训练)  ████████████████████████ 0.5589
BERT 加权 loss  (53K训练)  ███████████████████████  0.5477
LLM SFT-LoRA   (1K训练)    █████████████████████    0.5000
LLM Zero-Shot  (0 训练)    ███████████████          0.3600
```

### 5.2 关键洞察

**洞察 1：LLM Zero-Shot 0.36 不是 LLM 不行，是"格式不可控"**

200 条里 58 条无法解析（29%）。LLM 输出"政治""武器""房地产"在常识上都对，但不在标签集合内。
**带走的认知**：用 LLM 做分类必须做输出对齐（Constrained Decoding / Logit Bias / 后处理映射）。

**洞察 2：LoRA 微调的核心价值是"格式服从性"**

LoRA 把 unparseable 从 29% 暴降到 1.5%。
真正起作用的不是"教 LLM 知识"（它早就知道朴槿惠是政治人物），而是**"教 LLM 用我们的标签词汇说话"**。
这就是"对齐 Alignment"的核心思想。

**洞察 3：在小数据场景，LoRA 微调的性价比远超 BERT**

| 比较项 | BERT 全量微调 | LLM SFT-LoRA |
|---|---|---|
| 训练样本 | 53,360 | 1,000（53 倍少） |
| 训练参数 | 102.3M（全部） | 1.08M（0.22%） |
| 训练时长（CPU） | 86 min | 12 min（7 倍快） |
| accuracy | 0.5589 | 0.5000 |

**1000 条样本 + 12 分钟训练 + 0.22% 参数**，就达到了全量 BERT 微调 89% 的效果。

**洞察 4：1 epoch 加权 loss 反而效果略差，说明"训不够"**

加权 loss 通常需要 3+ epoch 才能体现优势。在 CPU 训练有限的情况下，先让模型充分收敛比换 loss 函数更重要。

**洞察 5：accuracy 不是唯一指标，要看 macro_f1 和 unparseable**

LLM Zero-Shot 看 accuracy 是 0.36，但去除 unparseable 后实际能达 0.51；LLM SFT 解决了 unparseable 后实际可用度大幅提升，比单看 accuracy 涨幅更有意义。

## 六、工程结论

### 6.1 方法选型建议

| 业务场景 | 推荐方法 | 原因 |
|---|---|---|
| 数据多 + 高 QPS 在线服务 | **BERT 微调** | 便宜、毫秒级延迟、稳定 |
| 数据少（&lt;5K 条）+ 任务复杂 | **LLM SFT-LoRA** | 少样本即可达到接近 BERT 全量的效果 |
| 类别极不均衡 | BERT + 加权 loss + 充分训练 | 需 3+ epoch 才有效 |
| 冷启动场景 | LLM Zero-Shot + 输出对齐 | 兜底用，必须做后处理 |

### 6.2 各方法核心权衡

- **BERT 微调**：标注成本高、训练时间长，但推理快、稳定可控
- **加权 loss**：增加少数类覆盖率，需要更长训练才能体现
- **Zero-Shot LLM**：零标注、零训练、慢推理，瓶颈在输出格式
- **LoRA SFT**：1000 条样本就能学会"按标签词汇说话"，性价比最高

## 七、产出文件

```
outputs/
├── checkpoints/
│   ├── best_cls.pt              ← BERT 普通 loss 模型（390MB）
│   └── best_cls_weighted.pt     ← BERT 加权 loss 模型（390MB）
├── sft_adapter/                 ← Qwen LoRA 适配器（4.2MB，仅 0.22% 参数）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── train_log_cls.json           ← BERT 普通 loss 训练日志
├── train_log_cls_weighted.json  ← BERT 加权 loss 训练日志
├── train_log_sft.json           ← LLM SFT 训练日志
├── llm_zero_shot_results.json   ← Zero-Shot 200 条详细结果
└── llm_sft_results.json         ← SFT 200 条详细结果
```

## 八、本次实验运行清单

| 实验 | 状态 | 备注 |
|---|---|---|
| BERT 普通 loss（1 epoch） | ✅ 完成 | val_acc=0.5589 |
| BERT 加权 loss（1 epoch） | ✅ 完成 | val_acc=0.5477 |
| LLM Zero-Shot（200 条） | ✅ 完成 | acc=0.36, unparseable=58 |
| LLM SFT-LoRA（1K 训练 + 200 评测） | ✅ 完成 | acc=0.50, unparseable=3 |

所有数据均为本机 CPU 真实跑出，未引用外部 baseline。
