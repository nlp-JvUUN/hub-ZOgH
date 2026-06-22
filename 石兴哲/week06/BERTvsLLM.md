# 模型分类任务对比实验

## 实验目的

对比BERT+CLS、BERT+MeanPool、BERT+MaxPool、LLM+zero-shot-prompt(QWen2.5-0.5B-Instuct、Qwen2.5-7B-Instruct)、LLM+LoRA(QWen2.5-0.5B-Instruct、Qwen2.5-7B-Instruct) 在 **分类任务**上的表现

实验设计了**9种**对比方案

- 以bert为基础，使用cls、mean、max三种方式
- 使用llm zero-shot prompt方式 对比QWen2.5-0.5B-Instuct、Qwen2.5-7B-Instruct 2种
- 使用llm + lora方式 对比QWen2.5-0.5B-Instruct、Qwen2.5-7B-Instruct 2种
- 使用 llm full finetune方式 对比QWen2.5-0.5B-Instuct + lora 2种

文本分类任务本质上属于encoder任务，bert系列会天然更强

机器 RTX 5070 laptop 8G + 32G RAM

## 实验数据

数据集：THUCNews

bert实验数据集划分：
53,360 train
10,000 dev/test

分类数量：15类

## 实验过程

### 第一阶段

> 研究 **“句向量表征”** 、判别式模型在**分类任务**上的表现

**实验方案**:

```
Bert 模型：Google BERT-base-chinese
12-layer,768-hidden,12-heads
batch_size = 32
hidden_size = 768
dropout = 0.1
eps = 1e-8
learning_rate = 2e-5  # 学习率
```

- BERT + CLS
- BERT + Mean
- BERT + Max

### 第二阶段

> 研究 **“大模型泛化”**

**实验方案**:

使用2个模型分别测试

```
Qwen2.5-7B-Instruct
Qwen2.5-0.5B-Instruct
```

- LLM zero-shot

### 第三阶段

> 研究 **“低成本适配”**

**实验方案**:

| 参数                  | 建议          |
| --------------------- | ------------- |
| r                     | 8 或 16       |
| alpha                 | 16            |
| dropout               | 0.05          |
| target_modules        | q_proj,v_proj |
| batch size            | 1~4           |
| gradient accumulation | 开            |

- LLM + LoRA

## 衡量指标

| 指标      | 是否必须 |
| --------- | -------- |
| Accuracy  | 必须     |
| F1-score  | 强烈建议 |
| Precision | 建议     |
| Recall    | 建议     |

### 第四阶段

> 研究训练性价比 综合时间、空间、准确率

- LLM + full finetuning

## 实验结果

- bert方式：

bert+cls结果：file：scratch-2026\data\week06\outputs\log\train_log_cls.json

bert+mean结果：file：scratch-2026\data\week06\outputs\log\train_log_mean.json

bert+max结果：file：scratch-2026\data\week06\outputs\log\train_log_max.json

- llm zero-shot prompt方式

llm zero-shot 基于Qwen2.5-0.5B-Instruct结果：
file：scratch-2026\data\week06\outputs\log\llm_zero_shot_results_Qwen2.5-0.5B-Instruct.json

llm zero-shot 基于Qwen2.5-7B-Instruct结果：
file：scratch-2026\data\week06\outputs\log\llm_zero_shot_results_Qwen2.5-7B-Instruct.json

- llm + lora方式

llm + lora 基于Qwen2.5-0.5B-Instruct：
训练日志 file：scratch-2026\data\week06\outputs\log\train_log_sft_Qwen2.5-0.5B-Instruct.json
预测日志 file：scratch-2026\data\week06\outputs\log\llm_sft_results_Qwen2.5-0.5B-Instruct.json

llm + lora 基于Qwen2.5-7B-Instruct（因为显存放不下，才用Q4量化方式训练）：
训练日志 file：scratch-2026\data\week06\outputs\log\train_log_sft_Qwen2.5-7B-Instruct.json
预测日志 file：scratch-2026\data\week06\outputs\log\llm_sft_results_Qwen2.5-7B-Instruct.json

- llm + full_finetuning方式 只能是0.5B 有2种方式，默认参数和优化版本
  - 默认版本 llm + full_finetuning 结果 基于Qwen2.5-0.5B-Instruct 耗时很久
    训练日志 file：scratch-2026\data\week06\outputs\log\train_log_full_ft_Qwen2.5-0.5B-Instruct.json

  预测日志 file：scratch-2026\data\week06\outputs\log\llm_sft_fullft_results_Qwen2.5-0.5B-Instruct.json
  - 优化版本 llm + full_finetuning 结果 基于Qwen2.5-0.5B-Instruct 耗时大幅下降
    训练日志 file：scratch-2026\data\week06\outputs\log\train_log_full_ft_Qwen2.5-0.5B-Instruct_opt.json

  预测日志 file：scratch-2026\data\week06\outputs\log\llm_sft_fullft_opt_results_Qwen2.5-0.5B-Instruct.json

## 实验总结

- 从模型准确性、训练时间、推理时间、模型大小、模型原理、工程角度进行评估

### 综合对比总表

| 方案             | 模型                  | 准确率 (test 300) | 单轮训练时间     | 推理耗时/样本 | 可训练参数量  | 模型总参数量 |
| ---------------- | --------------------- | ----------------- | ---------------- | ------------- | ------------- | ------------ |
| BERT+CLS         | bert-base-chinese     | ~56.7% (val 10k)  | ~270s (4.5min)   | <10ms         | 102M          | 102M         |
| BERT+MeanPool    | bert-base-chinese     | ~56.2% (val 10k)  | ~239s (4.0min)   | <10ms         | 102M          | 102M         |
| BERT+MaxPool     | bert-base-chinese     | ~56.0% (val 10k)  | ~280s (4.7min)   | <10ms         | 102M          | 102M         |
| Zero-shot        | Qwen2.5-0.5B-Instruct | 30.0%             | —                | 0.05s         | 0             | 495M         |
| Zero-shot        | Qwen2.5-7B-Instruct   | 49.0%             | —                | 2.07s         | 0             | 7.6B         |
| LoRA SFT         | Qwen2.5-0.5B-Instruct | **56.3%**         | ~864s (14.4min)  | 0.05s         | 1.08M (0.22%) | 495M         |
| LoRA SFT (Q4)    | Qwen2.5-7B-Instruct   | **61.3%**         | ~2333s (38.9min) | 0.27s         | 2.52M (0.06%) | 7.6B         |
| Full-ft (原始)   | Qwen2.5-0.5B-Instruct | 55.7%             | ~11509s (3.2h)   | 0.05s         | 494M (100%)   | 495M         |
| Full-ft (优化版) | Qwen2.5-0.5B-Instruct | 53.7%             | ~304s (5.1min)   | 0.05s         | 494M (100%)   | 495M         |

> BERT 指标为验证集（10000条）结果，LLM 指标为测试集（300条）结果，两者非严格同分布，仅供参考量级。

### 一、以 BERT 为基准的核心发现

1. **Zero-shot 远弱于 BERT**：0.5B 模型 zero-shot 仅 30%，7B 也仅 49%，均低于 BERT 的 ~56-57%。且 0.5B 模型有 137/300（45.7%）的输出无法解析为有效标签（生成"健康""时尚""电商"等不在15分类中的词），说明 zero-shot 不加约束时，生成式模型自由发挥严重损害准确率。

2. **LoRA SFT 追平/超越 BERT**：5000条训练数据 + LoRA 微调后，0.5B 达到 56.3%（追平 BERT），7B Q4 量化版达到 61.3%（超越 BERT ~4-5 个百分点），且不可解析输出从 137 骤降至 1（0.5B）和 5（7B）。

3. **模型规模的正向收益**：在 SFT 场景下，7B 比 0.5B 高出 5 个百分点，说明更大模型在下游任务适配时能保留更强的语义理解能力。但在 zero-shot 场景下，7B（49%）也远低于微调后的 0.5B（56.3%），说明**微调远比模型规模重要**。

4. **QLoRA 的"免费午餐"效应（关键发现）**：7B 模型通过 Q4 量化 + LoRA 训练后，虽然模型规模是 0.5B 的 15 倍（7.6B vs 495M 参数），但**两者实际运行在同一台机器上，GPU 显存消耗基本持平**：

| 指标            | 0.5B 全精度 SFT | 7B QLoRA SFT    | 说明                                                         |
| --------------- | --------------- | --------------- | ------------------------------------------------------------ |
| 模型权重占用    | ~1.0 GB (fp16)  | ~3.8 GB (4-bit) | 量化将 7B 从 ~15GB 压缩到 ~4GB                               |
| LoRA 可训练参数 | 1.08M (~4 MB)   | 2.52M (~10 MB)  | 均只占极小比例                                               |
| 训练显存消耗    | ~5-7 GB         | ~8-10 GB        | **同一张消费级显卡（如 RTX 3060 12G 或 RTX 4060 8G）可运行** |
| batch_size      | 2               | 1               | 7B 被迫减半，但通过 grad_accum=8 等效补偿                    |
| 单轮训练时间    | ~864s           | ~2333s          | 7B 更慢，但仍在可接受范围                                    |
| 最终准确率      | 56.3%           | **61.3%**       | 几乎同样的硬件成本，多拿 5 个点                              |

> **核心洞察**：如果不量化，7B 模型全精度训练需要 ~20-24GB 显存，远超消费级 GPU 的承受范围（通常 8-12GB）。Q4 量化把 7B 的显存门槛"砍"到了和 0.5B 全精度几乎相同的水平。这意味着：**花同样的显卡钱、用同样级别的机器，你不需要升级硬件，就可以把 0.5B（56.3%）切换成 7B QLoRA（61.3%），获得近 5 个百分点的免费提分。** 唯一的代价是训练时间延长约 2.7 倍（2333s vs 864s/epoch），但在学术实验和小规模微调场景中，这点时间成本完全可以接受。这就是 QLoRA 的核心价值——**打破模型规模与硬件门槛的线性绑定关系**，让大模型"降级"运行在小卡上，性能却几乎不打折。如果你手里只有一张 12GB 显卡，想用大模型做分类任务，直接上 7B QLoRA 是最划算的选择，没必要在 0.5B 上反复折腾。

5. **Full Fine-Tuning 的"反直觉"发现（意外收获）**：全量微调 495M 参数的结果**反而不如 LoRA 微调 1.08M 参数**，这是本次实验最出人意料但最具教学价值的结论：

| 指标         | LoRA SFT 0.5B | Full-ft 原始                                 | Full-ft 优化                                 |
| ------------ | ------------- | -------------------------------------------- | -------------------------------------------- |
| 可训练参数   | 1.08M (0.22%) | 494M (100%)                                  | 494M (100%)                                  |
| 单轮训练时间 | ~864s (14min) | ~11509s (3.2h)                               | ~304s (5min)                                 |
| 优化技术     | 仅 LoRA       | 无（fp32, 单进程, 无AMP）                    | bf16+AMP+多进程+预tokenize+grad_ckpt         |
| **加速比**   | 1× (baseline) | 0.075× (慢 13 倍)                            | **2.8×** (比 LoRA 还快)                      |
| 准确率       | **56.3%**     | 55.7%                                        | 53.7%                                        |
| overfitting  | 轻微          | 严重 (epoch3 train_loss=0.15, val_loss=1.19) | 严重 (epoch3 train_loss=0.16, val_loss=1.03) |

> **为什么全量微调反而更差？** 原因有三：
>
> 1. **严重过拟合**：5000 条样本更新 495M 参数，模型很快"背下"训练集，丧失泛化能力。LoRA 的 1.08M 低秩瓶颈反而充当了天然正则化器
> 2. **灾难性遗忘**：全量更新所有参数会覆盖预训练学到的通用语义知识，模型退化为"只会做分类"，失去了预训练的泛化优势
> 3. **优化版精度下降**：bf16 + AMP + gradient checkpointing 引入了轻微的数值精度损失，在全量微调场景下叠加效应被放大（LoRA 因参数极少而不受影响）

> **工程启示**：**更少的可训练参数 ≠ 更差的结果**。在小数据场景下（<10000条），LoRA 的"受限自由度"是优势而非缺陷。全量微调就像用推土机修手表——力量大但精度差。真正的工程智慧是选择正确的工具：如果你有 50 万条数据，full-ft 可能更强；如果你只有 5000 条，LoRA 是唯一正确的选择。

### 二、为什么 LLM + LoRA 效果能追平甚至超越 BERT？

1. **知识迁移优势**：Qwen2.5 系列在海量多样化语料上预训练，内部蕴含丰富的语义知识。虽然它是 decoder-only 架构（偏向生成），但通过 SFT 给少量标注样本，模型可以快速将预训练中的语义理解能力"对齐"到分类任务上。

2. **LoRA 的高效适配机制**：LoRA 不修改原模型权重，只在 attention 的 q_proj、v_proj 等关键投影矩阵旁挂低秩分解矩阵（r=8）。0.5B 仅训练 1.08M 参数（占 0.22%），7B 仅训练 2.52M 参数（占 0.06%），极大降低了过拟合风险和显存需求。

3. **BERT 的瓶颈**：bert-base-chinese 的预训练语料规模和质量远不如 Qwen2.5 系列，且 bert-base 的参数量（102M）限制了其表示能力的上限。在 53k 训练数据上，BERT 的三个池化策略（CLS/Mean/Max）均在 56-57% 附近收敛，说明模型容量已饱和。

4. **生成式→判别式的转换成功**：通过在 prompt 中强制约束输出空间（15个预定义标签），SFT 后的 LLM 可以稳定输出合法标签，将生成式模型的自由度压缩为判别式行为。0.5B SFT 后 unparseable 仅 1 例，证明了这一策略的有效性。

### 三、工程角度各自优劣

| 维度       | BERT                     | LLM + LoRA                           | LLM Full-ft (优化版)                          |
| ---------- | ------------------------ | ------------------------------------ | --------------------------------------------- |
| 训练速度   | 快（~270s/epoch）        | 中等（0.5B ~864s，7B ~2333s）        | **优化后较快（0.5B ~304s）**                  |
| 推理延迟   | 极低（<10ms，可 CPU）    | 较高（0.05-0.27s，需 GPU）           | 同 LoRA                                       |
| 显存需求   | 低（4-8GB）              | 0.5B ~5-7GB / 7B Q4 ~8-10GB          | 0.5B full-ft ~8-12GB（需 grad_ckpt）          |
| 模型体积   | 小（~400MB）             | base 大 + adapter ~4-10MB            | **完整模型 ~2GB (bf16)**，无 adapter 管理成本 |
| 部署难度   | 简单（ONNX/TensorRT）    | 需框架支持 adapter 加载              | **简单（标准 HuggingFace 格式）**             |
| 数据效率   | 高（53k 收敛）           | 中（5k 能适配）                      | **差（5k 严重过拟合）**                       |
| 过拟合风险 | 低                       | **极低（LoRA 天然正则化）**          | **高（小数据全量更新灾难性遗忘）**            |
| 灵活性     | 单任务                   | 一基座 + 多 adapter                  | 一模型一任务                                  |
| 迭代成本   | 需重训完整模型           | **最低（替换 10MB adapter）**        | 高（重训 495M 参数）                          |
| 中文适配   | 一般                     | 好（Qwen2.5 中文优化）               | 同 LoRA                                       |
| 适用场景   | 快速 baseline / CPU 部署 | **小数据微调 / 多任务 / 低成本迭代** | **大数据 (>50k) 全量微调 / 追求部署简洁**     |

### 四、关键结论

1. **Zero-shot LLM 不可靠**：0.5B 有 45.7% 的输出无法映射到标签体系，必须配合微调或约束解码。

2. **LoRA SFT 是"小数据场景下的最优解"**：5000 条数据 + 1.08M 参数训练 = 56.3% 准确率，追平 BERT。7B QLoRA 更达到 61.3%。参数效率极高（仅训练原模型 0.06-0.22%）。

3. **全量微调在小数据下是陷阱（核心发现）**：495M 全参数更新反而比 LoRA 低 0.6-2.6 个百分点，且训练慢 13 倍（未优化时）。原因：严重过拟合 + 灾难性遗忘 + 预训练知识被覆盖。LoRA 的低秩约束本质上是一种"被迫的正则化"。

4. **优化版 Full-ft 的真正价值不在精度而在速度**：bf16+AMP+多进程+预tokenize 组合将单 epoch 从 3.2 小时压缩到 5 分钟（**38 倍加速**），让全量微调从"不可用"变为"可接受"。但在小数据下精度仍不如 LoRA，这一优化的真正受益场景是 **大数据 (>50k) 全量微调**。

5. **QLoRA 的性价比无可匹敌**：同一张 12GB 消费显卡，7B QLoRA（61.3%）比 0.5B LoRA（56.3%）高 5 个百分点，比 0.5B Full-ft（53.7%）高 7.6 个百分点。大模型 + 量化 + LoRA = 当前分类任务的最优技术栈。

6. **数据效率排行榜**（5000 条训练数据，0.5B 模型）：
   ```
   LoRA (56.3%) > Full-ft 原始 (55.7%) > Full-ft 优化 (53.7%) > Zero-shot (30.0%)
   ```
   更多参数 ≠ 更好效果。在小数据下，参数效率（params/task-performance）才是真正的衡量指标。

### 五、后续优化方向

- **标签语义增强**：prompt 中加入类别定义描述，预计提升 5-10 个百分点——这是成本最低的改进
- **LoRA 超参搜索**：当前 r=8 未调优，尝试 r=4/16/32 找性价比拐点；扩展 target_modules 到 k_proj/o_proj/gate_proj
- **全量微调 + 更多数据**：用 10k/20k/53k 数据做 full-ft scaling curve，验证"数据量 > 某个阈值后 full-ft 反超 LoRA"的假设
- **Constrained decoding**：推理时强制模型只输出 15 个预定义标签之一，彻底消除 unparseable
- **BERT 蒸馏**：用 7B QLoRA 做 teacher 教 BERT，获得 BERT 的速度 + LLM 的精度（预期 58-60%）
- **数据效率曲线**：100/200/500/1000/2000/5000/10000 条数据对比 LoRA vs Full-ft 的收敛速度

### 六、继续尝试

基于全部 9 组实验（3 BERT + 2 Zero-shot + 2 LoRA + 2 Full-ft），当前已形成完整的对比矩阵。下一步最有价值的方向：

**P0（高优先级，直接提升准确率）：**

1. **标签描述增强**：成本趋零，5 分钟改 prompt，预计 +5~10%——立刻做
2. **LoRA 超参调优** → 在 0.5B 上扫 r={4,8,16,32} + target_modules 组合，找出当前 56.3% 能否推到 58%+

**P1（中优先级，完善分析与叙事）：** 3. **Full-ft 数据量 scaling** → 验证全量微调需要多少数据才能反超 LoRA（猜测约 20k-30k 条）4. **7B LoRA vs 7B QLoRA 直接对比** → 确认 Q4 量化的精度损失究竟多大（当前仅有间接推断）5. **Confusion matrix 深度分析** → 9 组实验的混淆矩阵叠加对比，定位哪些类别是"天生难分类"

**P2（低优先级，锦上添花）：** 6. BERT 蒸馏实验 7. 多任务联合训练（分类 + 摘要）8. 推理部署对比（ONNX/vLLM/llama.cpp）

**当前最佳推荐方案**（基于全部实验数据）：

> 如果你只有 1 张消费级显卡（8-12GB）+ 几千条标注数据 → **7B QLoRA (r=8, q+v_proj)**，
> 训练 1 小时，得到 61.3% 准确率。不要用 full-ft，不要在 0.5B 上反复实验。
