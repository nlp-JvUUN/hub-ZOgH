# EXTENSIONS.md — GRPO 实验扩展模块说明

本目录在原 GRPO 算术实验基础上新增了**可配置的扩展框架**，支持自定义难度、奖励权重和模型，
且新增的纯逻辑模块均可在**无 GPU 环境**下单测验证。

## 新增文件清单

| 文件 | 作用 | 依赖 GPU |
|------|------|:---:|
| `src/arithmetic_levels.py` | 难度题库扩展（新增 L7/L8/L9）+ 课程配比 | 否 |
| `src/rewards.py` | 参数化复合奖励框架（4 个可配置分量） | 否 |
| `src/train_grpo_configurable.py` | 全参数化 GRPO 训练脚本 | 是 |
| `src/test_levels.py` | 难度题库单测（纯 CPU） | 否 |
| `src/test_rewards.py` | 奖励函数单测（纯 CPU） | 否 |

## 1. 难度题库扩展（`arithmetic_levels.py`）

### 新增难度

| 级别 | 名称 | 题型示例 | 设计意图 |
|------|------|---------|---------|
| L7 | 带括号四则运算 | `(12 + 8) × 3 = 60` | 验证算术能力向新题型泛化 |
| L8 | 整除除法 | `144 ÷ 12 = 12` | 补全四则运算，整除保证可验证 |
| L9 | 混合四则运算 | `12 + 8 × 3 = 36` | 考察运算优先级，向真实数学题过渡 |

### 课程配比

```python
from arithmetic_levels import LEVEL_MIX_DEFAULT, LEVEL_MIX_EXTENDED, build_curriculum

# 默认配比（与原 train_grpo.py 一致）
# L3:0.50 / L5:0.25 / L2:0.25
rows = build_curriculum(LEVEL_MIX_DEFAULT, n=1000, seed=123)

# 扩展配比（含新难度）
# L3:0.30 / L5:0.25 / L7:0.20 / L8:0.15 / L2:0.10
rows = build_curriculum(LEVEL_MIX_EXTENDED, n=1000, seed=123)

# 自定义配比
custom = {"L7_paren_arith": 0.6, "L8_division": 0.4}
rows = build_curriculum(custom, n=500, seed=42)
```

**重要**：新增难度进训练集前，应先跑 `probe_baseline.py` 摸底其 `informative group rate`
（组内有对有错的比例），只在 0.3~0.8 区间的难度才适合 GRPO 训练。太易（全对）或太难（全错）
的组 advantage 为 0，不产生梯度。

### 向后兼容

`arithmetic_levels.make_problem` 完全兼容原 `probe_baseline.make_problem` 的 6 个难度，
行为一致。可直接替换：

```python
# 原 probe_baseline.py 顶部
from probe_baseline import make_problem
# 可改为（功能等价，且多了 L7/L8/L9）
from arithmetic_levels import make_problem
```

## 2. 奖励框架扩展（`rewards.py`）

### 四个可配置奖励分量

| 分量 | 默认权重 | 说明 |
|------|:---:|------|
| `correct` | 1.0 | 答案正确（宽松解析，与原版一致） |
| `format` | 0.2 | 输出含 `<answer>数字</answer>` |
| `cot_step` | 0.0 | 输出含推理步骤痕迹（`=数字`/`因此`/`Step`） |
| `length_penalty` | 0.0 | 长度惩罚（过短无推理/过长啰嗦都扣分） |

### 用法

```python
from rewards import RewardConfig, build_reward_funcs, compute_total_reward

# 默认配置 = 原版行为（correct 1.0 + format 0.2）
cfg = RewardConfig()

# 开启 CoT 过程奖励 + 长度惩罚
cfg = RewardConfig(weight_cot_step=0.3, weight_length_penalty=1.0)

# 构建 TRL 兼容的奖励函数列表
reward_funcs, names = build_reward_funcs(cfg)
# -> ([reward_correct, reward_format, reward_cot_step, reward_length_penalty],
#     ["reward_correct", "reward_format", "reward_cot_step", "reward_length_penalty"])

# 离线计算单条输出的奖励（无需 GPU，供分析用）
r = compute_total_reward("47 + 38 = 85，<answer>85</answer>", 85, cfg)
# -> {"correct": 1.0, "format": 0.2, "cot_step": 0.3, "length_penalty": 0.0, "total": 1.5}
```


- **权重设计**：强信号会稀释弱信号（见 ARCHITECTURE §3.2）。默认 `format=0.2` 故意小于
  `correct=1.0`，可观察"格式分收敛慢于正确分"的现象。
- **CoT 奖励**：演示过程奖励（PRM）思路。开启时需配合 `--max-completion-length` 加大
  （如 128），否则模型无空间输出推理步骤。
- **长度惩罚**：演示行为塑形。先摸清基线输出长度分布，再设 `length_ideal_min/max`。

## 3. 可配置训练脚本（`train_grpo_configurable.py`）

全参数化版本，与原 `train_grpo.py` 默认行为等价，但支持任意组合：

```bash
# 默认（= 原 train_grpo.py）
python src/train_grpo_configurable.py

# 换模型 + 扩展难度 + 开 CoT 奖励
python src/train_grpo_configurable.py \
    --model /path/to/Qwen2.5-1.5B-Instruct \
    --level-mix L3_addsub_3digit:0.3,L5_mul_2x1digit:0.25,L7_paren_arith:0.2,L8_division:0.15,L2_addsub_2digit:0.1 \
    --reward-cot-step 0.3 \
    --max-completion-length 128 \
    --tag ext

# LoRA + 调高温度保持多样性
python src/train_grpo_configurable.py --lora --temperature 1.2 --tag hot

# 只给正确分（消融：验证"RL 只优化被奖励的行为"，格式率应停在 0）
python src/train_grpo_configurable.py --reward-format 0.0 --tag nofmt
```

**关键参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | Qwen2-0.5B 路径 | 任意 Chat 模型路径 |
| `--level-mix` | 默认配比 | `L3:0.5,L5:0.25` 格式，或 `extended` |
| `--reward-correct` | 1.0 | 正确分权重 |
| `--reward-format` | 0.2 | 格式分权重 |
| `--reward-cot-step` | 0.0 | CoT 步骤奖励权重 |
| `--reward-length-penalty` | 0.0 | 长度惩罚权重 |
| `--num-generations` | 8 | 组内采样数 K |
| `--beta` | 0.0 | KL 系数（>0 会加载参考模型） |
| `--temperature` | 1.0 | 采样温度 |
| `--max-completion-length` | 64 | 补全最大长度 |

每次训练会额外输出 `outputs/train_config_{mode}{tag}.json` 记录完整配置，便于复现。

## 4. 单测运行（纯 CPU，本机可跑）

```bash
# 难度题库单测：9 个难度生成/求值/配比/分布
python src/test_levels.py

# 奖励函数单测：4 分量/工厂函数/边界情况
python src/test_rewards.py
```

两个单测均**不依赖 torch/trl/GPU**，只测试纯 Python 逻辑。当前已在本机验证全部通过。

## 5. 推荐的扩展实验

基于新框架可做的消融实验（按教学价值排序）：

1. **格式分权重扫描**：`--reward-format 0.0/0.2/0.5/1.0`，观察格式收敛速度与正确率爬坡
   的信号竞争（验证 ARCHITECTURE §3.2 的权重稀释现象）
2. **CoT 过程奖励**：`--reward-cot-step 0.3 --max-completion-length 128`，观察模型是否
   学会输出推理步骤，以及这对正确率的影响
3. **新增难度泛化**：用扩展配比训练，再在 L7/L8/L9 上评估，看训练集内 vs 外的泛化差异
4. **KL 防漂移**：`--beta 0.04`，观察是否抑制熵崩溃（原版 beta=0 熵降到 0.02~0.10）
5. **K 值影响**：`--num-generations 4/8/16`，观察组大小对 advantage 估计质量的影响

## 6. 环境要求

| 组件 | 单测 | 训练 |
|------|:---:|:---:|
| Python 3.10+ | ✓ | ✓ |
| torch | 可选（无也能跑） | 必须 + CUDA |
| trl 0.21 + transformers 5.x | 不需要 | 必须（配 trl_compat） |
| peft | 不需要 | 仅 `--lora` 时 |
| GPU | 不需要 | 必须（8GB+） |

单测仅用标准库 + 项目内模块，可在任何 Python 环境运行。
