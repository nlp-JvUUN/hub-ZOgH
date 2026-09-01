# GRPO Math Reasoning Experiment

这个小项目用于实验：用 GRPO 强化学习提升小语言模型解数学题的能力。

核心思路：

1. 选择一个可指令跟随的基础模型，例如 `Qwen/Qwen2.5-0.5B-Instruct` 或更大的同系列模型。
2. 用数学题数据集生成多条候选答案。
3. 用奖励函数打分：答案是否正确、是否按要求给出最终答案。
4. 用 TRL 的 `GRPOTrainer` 更新模型。
5. 在 GSM8K 等测试集上比较训练前后的准确率。

## 环境

建议在 CUDA/Linux 环境跑训练；Windows 可以先跑奖励函数和小规模评测。先安装与你显卡匹配的 PyTorch，再安装项目依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

如果没有 NVIDIA GPU，可以先用 `--max-steps 1 --max-train-samples 8` 做流程验证，但真实 GRPO 训练会很慢。

## 快速开始

先测试奖励函数：

```bash
python scripts/smoke_test_rewards.py
```

训练一个很小的 GRPO LoRA 实验：

```bash
accelerate launch scripts/train_grpo_math.py ^
  --model-name Qwen/Qwen2.5-0.5B-Instruct ^
  --dataset-name trl-lib/DeepMath-103K ^
  --output-dir outputs/qwen25-05b-grpo-math ^
  --max-steps 200 ^
  --per-device-train-batch-size 1 ^
  --gradient-accumulation-steps 8 ^
  --num-generations 4 ^
  --use-lora
```

评测基础模型：

```bash
python scripts/evaluate_math.py ^
  --model-name Qwen/Qwen2.5-0.5B-Instruct ^
  --dataset-name gsm8k ^
  --dataset-config main ^
  --split test ^
  --limit 200
```

评测 GRPO LoRA checkpoint：

```bash
python scripts/evaluate_math.py ^
  --model-name Qwen/Qwen2.5-0.5B-Instruct ^
  --adapter-path outputs/qwen25-05b-grpo-math/checkpoint-200 ^
  --dataset-name gsm8k ^
  --dataset-config main ^
  --split test ^
  --limit 200
```

## 推荐实验记录

建议每次实验记录这些字段：

- 基础模型
- 数据集与样本数
- `num_generations`
- `max_completion_length`
- reward 组成与权重
- 是否 LoRA、LoRA rank
- 训练步数
- GSM8K 子集准确率
- 失败样例：错误答案、格式错误、推理截断

## 文件说明

- `scripts/train_grpo_math.py`：GRPO 训练入口。
- `scripts/evaluate_math.py`：数学题准确率评测。
- `scripts/smoke_test_rewards.py`：奖励函数冒烟测试。
- `src/grpo_math/rewards.py`：答案抽取、正确性奖励、格式奖励。
- `configs/accelerate_single_gpu.yaml`：单卡训练的 Accelerate 配置示例。

## 预期现象

小规模训练通常先提升“答案格式”和简单算术题稳定性；复杂推理能力提升需要更多样本、更长训练和更强基础模型。GRPO 对奖励函数很敏感，如果模型学会只迎合格式但准确率不升，需要降低格式奖励权重或加强答案验证。
