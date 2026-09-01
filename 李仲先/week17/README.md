# 基于 GRPO 的强化学习提升 Qwen2.5-0.5B-Instruct 数学能力

使用 TRL 的 GRPO(Group Relative Policy Optimization)算法,在 GSM8K 数学题上对本地
`Qwen2.5-0.5B-Instruct` 模型做 LoRA 强化学习微调,并通过训练前后在 GSM8K 测试集上的
准确率对比验证数学能力提升。

## 原理

GRPO(DeepSeekMath 论文提出)对每个提示词采样一组回答(本实验 G=8),用奖励函数给每个
回答打分,在组内做标准化得到优势(advantage),再用 PPO 式裁剪目标更新策略,并带
KL 正则(beta 项)约束模型不偏离参考模型太远,无需单独训练价值网络。

奖励函数(加权求和):

| 奖励 | 权重 | 说明 |
| --- | --- | --- |
| accuracy_reward | 1.0 | 用 math_verify 校验 `\boxed{}` 内答案与标准答案是否等价 |
| think_format_reward | 0.2 | 推理过程是否包在 `&lt;think&gt;...&lt;/think&gt;` 标签中 |
| boxed_format_reward | 0.2 | 是否给出 `\boxed{}` 格式的最终答案 |

## 文件

- `prepare_data.py` — 下载 GSM8K,转成对话格式(`prompt` + `solution` 列),生成 `data/*.parquet`
- `train_grpo.py` — GRPO 训练脚本(TRL GRPOTrainer + LoRA)
- `eval_math.py` — 在 GSM8K 测试集上贪心解码评测准确率,支持加载 LoRA 适配器
- `run.sh` — 一键运行:准备数据 → 基线评测 → GRPO 训练 → 训练后评测

## 环境

```
pip install torch transformers trl peft datasets accelerate math_verify latex2sympy2_extended
# triton 编译需要 C 编译器(无系统 gcc 时):
conda install -c conda-forge gcc_linux-64 gxx_linux-64
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
```

## 运行

```bash
bash run.sh
```

单独运行各步骤:

```bash
python prepare_data.py
python train_grpo.py --max_steps 50 --num_generations 8 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --learning_rate 5e-6
python eval_math.py                                        # 基线
python eval_math.py --adapter_path output/grpo-qwen-gsm8k  # 训练后
```

## 主要超参数

| 参数 | 值 |
| --- | --- |
| num_generations (G) | 8 |
| max_completion_length | 1024 |
| temperature | 1.0 |
| beta (KL) | 0.04 |
| learning_rate | 5e-6 (cosine, warmup 5 步) |
| LoRA r / alpha | 16 / 32 (作用于 q/k/v/o/gate/up/down_proj) |
| 有效批大小 | 2 提示词 × 8 采样 × 4 梯度累积 |

## 结果

GSM8K 测试集 200 题,贪心解码:

| 指标 | 训练前 | GRPO 训练后 |
| --- | --- | --- |
| 准确率 | 41.5% (83/200) | **43.0% (86/200)** |
| `\boxed{}` 格式使用率 | 61.5% | **82.0%** |
| 平均回答长度(字符) | 912 | 948 |

训练 50 步(每步 2 提示词 × 8 采样),共约 100 个不同提示词。准确率提升 +1.5 个百分点
(答对新增 16 题,遗忘 13 题),输出格式遵循率大幅提升 +20.5 个百分点。继续增加训练步数
(如 `--max_steps 300`)可获得更明显的提升。

详细结果见 `output/eval_before.json` 与 `output/eval_after.json`。
