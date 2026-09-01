# GRPO 强化学习提升模型做数学题能力 — 独立实验项目

> 本周作业：实验基于 GRPO 的强化学习提升模型做数学题能力。
> 本目录是一套**独立编写、可提交**的最小实现，不依赖 TRL，只用 torch + transformers，
> 可在纯 CPU 环境跑通并产出真实的验证产物。

## 1. 项目定位

用 **GRPO（Group Relative Policy Optimization）** 在 **Qwen2-0.5B-Instruct** 上做算术题 RL。
目标是让模型同时学会两件事：

1. **输出格式**：把答案放进 `<answer>N</answer>`（格式分）。
2. **算得更准**：答案正确（正确分）。

奖励完全由**程序规则**判定（可验证奖励 RLVR），无需奖励模型、无需人工标注，
是教学级 RL 最干净、最可复现的载体。

## 2. 目录结构

```
week17_grpo_homework/
├── src/
│   ├── reward.py      # 题目生成（6 难度）、输出解析、复合奖励（correct=1.0 + format=0.2）
│   ├── grpo_loop.py   # 自实现最小 GRPO：采样→组内归一化 advantage→PPO-clip surrogate→更新
│   ├── probe.py       # 基线摸底 / 训练后评估（greedy + pass@8 + informative group rate）
│   ├── train.py       # 主训练脚本（CLI：--max_steps / --micro_batch / --lr / --seed）
│   └── analyze.py     # 训练前后对比表 + 训练曲线图（matplotlib 不可用时降级文本表）
├── outputs/           # 运行产物（JSON 指标 / checkpoint / 曲线图）
├── requirements.txt
└── README.md          # 本文件
```

## 3. 环境准备

```bash
pip install -r requirements.txt
# torch 与 transformers 必需；matplotlib 可选（仅用于 analyze.py 画曲线）
```

- 可选的 `transformers` 版本建议 ≥ 4.40；当前在 `transformers 5.13.0` + `torch 2.13.0+cpu` 上验证通过。
- **模型路径**：默认 `D:\八斗学习内容\pretrain_models\Qwen2-0.5B-Instruct`（可用 `--model` 覆盖）。
  代码会按 `torch.cuda.is_available()` 自动选设备：有 GPU 用 cuda，无 GPU 用 cpu。

## 4. 运行流程

```bash
# Step 0：进入项目根目录（以下假定已在 week17_grpo_homework 下）
cd week17_grpo_homework

# Step 1：基线摸底（定难度）。--n 控制每难度题数，CPU 上建议 6~10
python src/probe.py --n 6 --out outputs/baseline_probe.json

# Step 2：GRPO 训练（CPU 上默认 12 步，约 12-20 分钟；--max_steps 可调）
python src/train.py --max_steps 12 --out outputs/train_log.json
# Step 3：训练后评估（必须与基线同 seed，才能配对比较）
python src/probe.py --n 6 --model outputs/ckpt --seed 42 --out outputs/post_probe.json

# Step 4：对比表 + 训练曲线
python src/analyze.py
```

冒烟测试（快速确认流程可跑）：
```bash
python src/probe.py --quick --out outputs/baseline_probe.json      # 每难度 10 题
python src/train.py --max_steps 2 --micro_batch 1 --quick          # 只跑 2 步、每步 1 prompt
```

关键 CLI：`--max_steps`（优化步数）、`--micro_batch`（每步 prompt 数）、`--epochs`（每步对采样做几次更新）、
`--lr`（默认 2e-6；**无 KL/参考模型的纯 RL 必须低学习率**，调到 3e-5 会发散并摧毁模型）、
`--num_generations`（组内采样数 K=8）、`--seed`（默认 42）。

## 5. 关键设计

### 5.1 为什么用算术题
| 维度 | 算术题优势 |
|------|-----------|
| 奖励可验证 | 答案判定精确、零成本、零噪声 |
| 难度可控 | 位数/运算即难度旋钮，可精确调到 GRPO 可学习区间 |
| 叙事门槛低 | 与 DeepSeek-R1 / TinyZero 同款的"可验证奖励 RL"故事 |
| 资源友好 | 答案只有几个 token，生成快，CPU 也能跑 |

### 5.2 GRPO 核心（自实现，见 `grpo_loop.py`）
对每个 prompt 采样 K=8 条 completion，得组内奖励 `r_i`：
```
advantage A_i = (r_i − mean(r)) / (std(r) + eps)     ← 组内 z-score 归一化，替代 Critic 价值网络
ratio       = exp(log π_θ(c_i) − log π_old(c_i))     ← 新旧策略概率比
surrogate   = min(ratio·A_i, clip(ratio,1−ε,1+ε)·A_i)  ← PPO-clip（ε=0.2，beta=0 无 KL 项）
loss        = −mean(surrogate)
```
- 无价值网络、无奖励模型、无参考模型（beta=0），只用组内奖励方差出 advantage。
- **组内全对/全错时 advantage=0，该组不产生梯度**——这正是"选题难度"是 GRPO 工程核心的原因。
- 每个优化步采样一次，然后对这批采样做 `--epochs`（默认 4）次梯度更新：
  每次用当前策略重算新 `log π_θ`，与采样时冻结的 `log π_old` 求 `ratio`，`ratio ≠ 1` 时 clip 真正生效。

### 5.3 复合奖励（`reward.py`）
```
reward = reward_correct(1.0 或 0.0)   # 宽松解析：有 <answer> 取标签内数字，否则取最后一个数字
       + reward_format(0.2 或 0.0)     # 输出含 <answer>N</answer> 即得分
```
- 正确分用**宽松解析**，保证冷启动（模型还不会输出标签）时正确信号不为 0，训练能启动。
- 格式分权重 0.2 故意小于正确分 1.0，用于观察"主次信号竞争"的真实 RL 现象。

### 5.4 难度选题（informative group rate）
GRPO 的梯度来自**组内奖励方差**。选题核心指标是 **informative group rate**
（组内 0 < 正确数 < K 的比例）。把 L3 / L5 / L2 选进训练集（这些难度 informative 较高），
把 L1 / L4 / L6 留作**未训练难度的泛化对照**。

## 6. 产物解读

- `outputs/baseline_probe.json`：基线各难度 greedy/采样/pass@8/格式率/informative rate。
- `outputs/train_log.json`：每步 loss / surrogate / reward_mean / entropy。
- `outputs/post_probe.json`：训练后评估（与基线同 seed，可配对比较）。
- `outputs/figures/curves.png`：训练曲线（loss / reward / entropy）。
- `outputs/ckpt/`：训练后的模型 checkpoint。

健康训练标志：**entropy 下降 + reward_mean 上升**（策略在目标任务上趋于收敛）。
若 reward 恒为 0 或 loss 出现 NaN，参见《踩坑排查》一节。

## 7. 踩坑排查（本机实测）

| 现象 | 根因与解决 |
|------|-----------|
| `device_map="auto"` 报错要 accelerate | 改为直接 `.to(device)`；单设备无需 accelerate |
| `generate` 报 `'<' not supported between str and int` | `pad_token_id` 必须传 **int**（如 `tokenizer.pad_token_id`），不要传 token 字符串 |
| 训练一步后输出变乱码/`!!!!!` | 模型可能被 fp16 加载训废；始终显式 `dtype=torch.bfloat16` |
| 组内全对/全错`std=0` | advantage 分母加 `eps=1e-8`；该组梯度为 0 属 GRPO 预期，非 bug |
| CPU 训练慢 | 降 `--max_steps` / `--micro_batch`；生成是主要瓶颈 |

## 8. 扩展方向（可做消融）
- 把格式分权重 0.2 调成 1.0，观察"格式秒收敛、正确率爬坡变慢"。
- 去掉 `reward_format`，看格式率是否停在 0（验证"RL 只优化被奖励的行为"）。
- 把 L6（两位数×两位数）加入训练集，观察 informative rate 塌掉后学习效率下降。
- 调 `K`（4 / 8 / 16）看组大小对 advantage 估计质量的影响。
