# 基于 GRPO 的强化学习提升模型做数学题能力

这是一个可直接作为课程作业提交的完整实验项目，实现了：

1. 六档算术题程序化生成；
2. 基座模型 greedy 与 pass@k 基线评估；
3. 基于 TRL `GRPOTrainer` 的强化学习；
4. 正确性奖励 + 格式奖励的复合奖励塑形；
5. 训练前后准确率、格式率和泛化能力分析；
6. 附带实际实验 JSON、训练日志和图表。

详细原理、实验过程和结果见 **[实验报告.md](实验报告.md)**。

## 目录结构

```text
.
├─ README.md
├─ 实验报告.md
├─ requirements.txt
├─ run_experiment.bat
├─ src/
│  ├─ math_utils.py          # 题目生成、提示词和答案解析
│  ├─ evaluate.py            # 统一评估
│  ├─ train_grpo.py          # GRPO/LoRA 训练
│  ├─ analyze_results.py     # 汇总结果并绘图
│  └─ trl_compat.py          # 旧版 TRL 兼容补丁
└─ outputs/
   ├─ baseline_results.json  # 训练前真实结果
   ├─ grpo_results.json      # 训练后真实结果
   ├─ train_log.json         # 200 步训练日志
   └─ figures/
```

## 快速查看已有结果

不需要 GPU，只需安装 matplotlib：

```bash
python src/analyze_results.py
```

该命令会打印训练前后对比表，并生成：

```text
outputs/figures/accuracy_comparison.png
```

## 复现实验

推荐 Python 3.10/3.11 和 NVIDIA CUDA GPU。

```bash
pip install -r requirements.txt
```

默认模型为公开模型 `Qwen/Qwen2.5-0.5B-Instruct`。也可使用本地模型路径：

```bash
# 快速检查评估流程
python src/evaluate.py --model D:\path\to\model --quick \
  --output outputs/baseline_new.json

# 3 步冒烟训练
python src/train_grpo.py --model D:\path\to\model --max-steps 3 \
  --output-dir outputs/grpo_model_smoke

# 正式训练
python src/train_grpo.py --model D:\path\to\model --max-steps 200 \
  --output-dir outputs/grpo_model

# 显存不足时启用 LoRA
python src/train_grpo.py --model D:\path\to\model --lora \
  --output-dir outputs/grpo_model

# 训练后评估
python src/evaluate.py --model outputs/grpo_model \
  --output outputs/grpo_new.json
```

Windows 用户可以修改 `run_experiment.bat` 中的 `MODEL`，然后双击执行完整流程。

## 主要实验结果

每个难度使用 50 道固定种子的测试题：

| 指标 | 基线 | GRPO 后 |
|---|---:|---:|
| 六档宏平均 greedy 正确率 | 49.67% | 83.33% |
| 训练难度 L2/L3/L5 平均正确率 | 45.33% | 92.00% |
| 未训练难度 L1/L4/L6 平均正确率 | 54.00% | 74.67% |
| 六档宏平均格式率 | 1.33% | 95.00% |

其中 L5（两位数乘一位数）从 20% 上升到 88%，绝对提升 68 个百分点。

## 注意事项

- 本项目不附带大型全量模型权重，避免作业目录体积过大；可按命令自行训练。
- `outputs` 中附带的是实际实验指标和训练日志，可直接用于报告核验。
- `trl_compat.py` 仅用于某些旧版 TRL/Transformers 组合；使用兼容的新版本时通常不会触发补丁。
- 正式训练需要 CUDA GPU；只查看报告和运行分析脚本不需要 GPU。
