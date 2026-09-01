# 本周作业报告：基于 GRPO 的强化学习提升模型做数学题能力

> 独立实现一份可提交的 GRPO 强化学习代码，并在当前无 GPU 的 CPU 环境真实跑出验证产物。
> 本报告与项目代码位于 `week17_grpo_homework/`，与课程目录下已有的 `grpo_arithmetic` **完全独立**（不复用其任何源代码或产物）。

## 1. 作业目标与实验设计

用 **GRPO（Group Relative Policy Optimization）** 在 **Qwen2-0.5B-Instruct** 上做算术题强化学习，
让模型同时学会两件事，并量化对比训练前后差异：

1. **输出格式**：把答案放进 `<answer>N</answer>` 标签。
2. **算得更准**：答案正确。

奖励完全由**程序规则**判定（可验证奖励 RLVR），无需奖励模型、无需人工标注，是教学级 RL 最干净、最可复现的载体。

**为什么用算术题**：

| 维度 | 算术题优势 |
|------|-----------|
| 奖励可验证 | 答案判定精确、零成本、零噪声 |
| 难度可控 | 位数/运算即难度旋钮，可精确调到 GRPO 可学习区间 |
| 叙事门槛低 | 与 DeepSeek-R1 / TinyZero 同款的可验证奖励 RL 故事 |
| 资源友好 | 答案只有几个 token，生成快，CPU 也能跑通 |

## 2. 独立实现的方法（不依赖 TRL）

当前环境为 `transformers 5.13.0`（比 `trl 0.21` 所适配的 `5.5.3` 更新），两者有版本兼容地雷（trl 0.21 依赖 transformers 4.x/5.x 的私有 API）。为保证**独立、可提交、可解释**，本作业用纯 torch 从零实现最小 GRPO：

```
奖励    reward = reward_correct(1.0 正确 / 0.0 错误，宽松解析)
              + reward_format(0.2 输出含 <answer>N</answer>)

GRPO    advantage A_i = (r_i - mean(r)) / (std(r) + eps)      ← 组内 z-score 归一化，替代 Critic
        ratio_i      = exp(log π_θ(c_i) - log π_old(c_i))     ← 新旧策略概率比
        surrogate    = min(ratio_i·A_i, clip(ratio_i, 1-ε, 1+ε)·A_i)   ← PPO-clip（ε=0.2, beta=0 无 KL）
        loss         = -mean(surrogate)
```

关键点：
- 每个优化步**采样一次**（旧策略冻结 `log π_old`），然后对这批采样做 **`--epochs` 次梯度更新**——
  每次用当前策略重算 `log π_θ`，`ratio ≠ 1` 时 clip 真正生效（这是 GRPO/PPO 区别于朴素策略梯度的核心）。
- **组内全对/全错时 advantage=0，该组不产生梯度**——选题难度是 GRPO 工程的核心。

### 代码结构（全新编写）
```
week17_grpo_homework/
├── src/
│   ├── reward.py      # 6 难度题目生成、输出解析、复合奖励（correct=1.0 + format=0.2）
│   ├── grpo_loop.py   # 自实现最小 GRPO：采样→组内归一化 advantage→PPO-clip surrogate→多 epoch 更新
│   ├── probe.py       # 基线摸底 / 训练后评估（greedy + pass@8 + informative group rate）
│   ├── train.py       # 主训练脚本（CLI：--max_steps/--micro_batch/--epochs/--lr/--seed）
│   └── analyze.py     # 训练前后对比表 + 训练曲线图
├── outputs/           # 本机真实运行产物
├── requirements.txt   # torch / transformers / matplotlib(可选)
└── README.md          # 运行说明与设计文档
```

## 3. 运行环境与配置

| 项目 | 值 |
|------|-----|
| 硬件 | **CPU 无 GPU**（torch `2.13.0+cpu`，本机 4 线程） |
| 模型 | Qwen2-0.5B-Instruct（`D:\八斗学习内容\pretrain_models\Qwen2-0.5B-Instruct`，494M 参数，bf16 加载） |
| 依赖 | torch、transformers 5.13.0、matplotlib（本机补装） |
| GRPO 超参 | K=8、ε=0.2、temperature=1.0、max_new_tokens=64、epochs=4、**lr=2e-6** |
| 训练规模 | 8 步 × 2 prompt/步 × 8 采样；训练集难度配比 L3 50% / L5 25% / L2 25% |
| 评估 | 6 难度 × 6 题 × (greedy + 8 采样)，seed=42（与基线配对可比） |

## 4. 实验结果（本机真实数据）

### 4.1 训练前后对比表（同一评估集，格式率 / greedy正确率 / pass@8）

| 难度 | 在训练集 | baseline | 训练后 |
|------|:---:|---------|--------|
| L1 个位数加法 | — | 0.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** |
| L2 两位数加减 | √ | 0.00 / 0.33 / 0.67 | **1.00 / 1.00 / 1.00** |
| L3 三位数加减 | √ | 0.33 / 0.50 / 0.83 | **1.00 / 0.67 / 1.00** |
| L4 表内乘法 | — | 0.00 / 0.67 / 0.83 | **1.00 / 1.00 / 1.00** |
| L5 两位×一位 | √ | 0.17 / 0.33 / 0.67 | **1.00 / 0.83 / 0.83** |
| L6 两位×两位 | — | 0.17 / 0.00 / 0.00 | **1.00 / 0.00 / 0.00** |

### 4.2 样例对照（greedy 解码）

| 题目 | 基线 | 训练后 |
|------|------|--------|
| 87 − 13 | `'64'`（错） | `'<answer>74</answer>'`（对+格式） |
| 56 × 4 | `'128'`（错） | `'<answer>224</answer>'`（对+格式） |
| 21 + 37 | `'58'`（无格式） | `'<answer>58</answer>'`（对+格式） |

### 4.3 训练曲线（`outputs/figures/curves.png`）

- **Reward（组均值）**：从 0.06 升到 ~0.4–0.76（第 7 步峰值 0.76），学习信号明显。
- **GRPO loss**：围绕 0 波动（组内归一化使 surrogate 均值≈0，非线性但 clip 在生效）。
- **Policy entropy**：稳定在 1.98–2.01，**未发散**。

## 5. 结果解读

1. **格式学习几乎完美且完全泛化**：所有难度（含未训练的 L1/L4/L6）格式率都到 **1.00**。
   格式是"表层行为"，RL 非常容易学会——与参考教学项目结论一致。
2. **训练集内难度正确率提升**：L2 0.33→1.00、L5 0.33→0.83（L3 因样本少且数值偏难，0.50→0.67）。
3. **未训练难度也在涨（能力泛化）**：L4 没进训练集，正确率 0.67→1.00；
   证明 RL 不只是背题，而是强化了"做对算术"的内部能力。
4. **超出能力边界的 L6 几乎不涨**：0.00→0.00（pass@8 也仅 0.00，且训后模型已学会输出 `<answer>` 格式，
   只是算错，如 `19×87` 输出 `1533` 而非 `1653`）。这是 RL 教科书级结论：
   **RL 是在模型能力边界内重排概率分布，不能凭空创造能力**。

## 6. 工程关键：稳定性与学习率（含一次真实发散实验）

本机实验记录了一个重要现象：**全量参数、无 KL/参考模型的纯 RL 必须用低学习率（2e-6）**。

| 尝试 | lr | epochs | 结果 |
|------|------|:---:|------|
| 单次更新/步 | 2e-6 | 1 | 稳定但 ratio≈1、loss≈0，几乎学不动（旧策略在采样与更新间变化太小，clip 不生效） |
| 多 epoch/步 | **3e-5** | 2 | **发散**：reward 崩到 0、熵从 2.29 升到 2.78，模型退化为输出长中文解释、丢失任务（全部难度正确率=0） |
| 多 epoch/步 | **2e-6** | 4 | **稳定且有效**：reward 0.06→0.76、熵稳定 1.98–2.01，格式率全难度到 1.00 |

**结论**：多 epoch（冻结旧策略 log-prob）让 clip 真正绑定，是 GRPO 显示学习效果的关键；
而 `lr=3e-5` 对无 KL 约束的全量 RL 会摧毁模型。这正是参考项目 `beta=0` 必须搭配 `2e-6` 低学习率的工程原因，
也是本项目独立复现得到的实证发现。

## 7. 可复现性与验证产物清单

所有产物均为**本机 CPU 真实运行生成**：

| 产物 | 路径 |
|------|------|
| 基线指标（6 难度 × 6 题） | `outputs/baseline_probe.json` |
| 训练日志（8 步，含 loss/reward/entropy） | `outputs/train_log.json` |
| 训练后评估（与基线同 seed 配对） | `outputs/post_probe.json` |
| 训练曲线图（loss / reward / entropy） | `outputs/figures/curves.png` |
| 训练后模型 checkpoint（988MB） | `outputs/ckpt/` |

复现命令（无需 TRL，纯 torch + transformers）：
```bash
cd week17_grpo_homework
python src/probe.py --n 6 --out outputs/baseline_probe.json          # 基线（seed=42）
python src/train.py --max_steps 8 --micro_batch 2 --epochs 4 --lr 2e-6  # 训练
python src/probe.py --n 6 --model outputs/ckpt --seed 42 --out outputs/post_probe.json  # 训练后评估
python src/analyze.py                                                # 对比表 + 曲线
```

## 8. 结论

- 独立实现了一套**不依赖 TRL、纯 torch 的最小 GRPO**，在无 GPU 的 CPU 环境端到端跑通，
  从基线摸底 → 训练 → 训练后评估 → 对比分析，产物齐全且可复现。
- 训练后：**格式遵循率全难度到 1.00**（零开始），正确率在训练难度（L2、L5）和未训练难度（L4）上明显提升，
  L5 从 0.33→0.83；而超能力边界的 L6 保持 0.00。
- 独立复现并实证了 RL 的两条核心结论：
  1. **可验证奖励 RL 能显著改善格式遵循与能力边界内的正确率**；
  2. **RL 无法突破模型能力边界**（L6 多步乘法），且**无 KL 约束的全量 RL 必须低学习率**，否则发散。

## 附：扩展方向
- 把格式分权重 0.2 调成 1.0，观察"格式秒收敛、正确率爬坡变慢"。
- 加入 L6 到训练集，观察 informative rate 塌掉后学习效率下降。
- 调 K（4/8/16），对比组大小对 advantage 估计质量的影响。
- 在有 GPU 的机器上加大 `--max_steps`，观察更长训练下的收敛与熵崩溃风险。
