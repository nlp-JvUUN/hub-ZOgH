"""
参数化复合奖励：在原"正确分 1.0 + 格式分 0.2"基础上支持可配置权重与新增奖励分量

设计原则：
  1. 与原 train_grpo.py 的 reward_correct / reward_format 行为一致（默认权重）
  2. 通过 RewardConfig 数据类统一管理权重，训练脚本传一次即可
  3. 新增两个可选奖励分量：
     - cot_step_reward：思维链步骤奖励（输出含 "步骤/Step/=" 等推理痕迹时给小分）
     - length_penalty：长度惩罚（过短=无推理、过长=啰嗦，都扣分）
  4. 所有奖励函数纯函数，可脱离 GPU 单测

奖励分量清单（最终 reward = 各分量加权和）：
  correct     1.0  答案正确（宽松解析，与原版一致）
  format      0.2  输出含 <answer>数字</answer>
  cot_step    0.0  输出含推理步骤痕迹（默认关，需手动开）
  length_pen  0.0  长度惩罚（默认关，需手动开）

教学点：
  - 复合奖励的权重设计是 reward shaping 的核心：强信号会稀释弱信号（见 ARCHITECTURE §3.2）
  - cot_step_reward 演示"过程奖励"思路：奖励推理过程而非只看结果
  - length_penalty 演示"行为塑形"：通过奖励形状引导输出长度分布
"""
import re
from dataclasses import dataclass, field
from typing import List

# 复用 arithmetic_levels 的解析逻辑
from arithmetic_levels import parse_output

TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")
# 推理步骤痕迹：常见的中英文推理标记
COT_PATTERNS = [
    re.compile(r"步骤|step\s*\d", re.IGNORECASE),
    re.compile(r"因此|所以|thus|therefore|so\s", re.IGNORECASE),
    re.compile(r"=\s*\d"),          # 中间计算式 = 数字
    re.compile(r"\d+\s*[+\-×÷]\s*\d+\s*="),  # a + b = 形式
]


@dataclass
class RewardConfig:
    """复合奖励配置。训练时实例化一次，reward 工厂闭包捕获它。"""
    weight_correct: float = 1.0
    weight_format: float = 0.2
    weight_cot_step: float = 0.0    # 默认关：过程奖励需配合 max_completion_length 加大
    weight_length_penalty: float = 0.0  # 默认关：长度惩罚需先摸清输出长度分布
    # 长度惩罚的合理区间（token 数近似用字符数估算）
    length_ideal_min: int = 5       # 理想最短：<answer>42</answer> 约 15 字符
    length_ideal_max: int = 80      # 理想最长：含简短推理
    length_penalty_strength: float = 0.1  # 每偏离理想区间的惩罚量


def make_reward_correct(config: RewardConfig):
    """工厂：返回闭包形式的 reward_correct，兼容 TRL 的 reward_func 签名。"""
    w = config.weight_correct
    def reward_correct(completions, answer, **kwargs):
        rewards = []
        for comp, ans in zip(completions, answer):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            ok = parse_output(text, int(ans))[2]
            rewards.append(w if ok else 0.0)
        return rewards
    reward_correct.__name__ = "reward_correct"
    return reward_correct


def make_reward_format(config: RewardConfig):
    """工厂：格式奖励。输出含 <answer>数字</answer> 即得分。"""
    w = config.weight_format
    def reward_format(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            ok = parse_output(text, 0)[0]
            rewards.append(w if ok else 0.0)
        return rewards
    reward_format.__name__ = "reward_format"
    return reward_format


def make_reward_cot_step(config: RewardConfig):
    """
    工厂：思维链步骤奖励。
    输出命中任一推理模式即给分。演示过程奖励（PRM 思路的简化版）。
    """
    w = config.weight_cot_step
    def reward_cot_step(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            hit = any(p.search(text) for p in COT_PATTERNS)
            rewards.append(w if hit else 0.0)
        return rewards
    reward_cot_step.__name__ = "reward_cot_step"
    return reward_cot_step


def make_reward_length_penalty(config: RewardConfig):
    """
    工厂：长度惩罚。
    过短（无推理）或过长（啰嗦/重复）都扣分；落在理想区间内不奖不罚。
    """
    w = config.weight_length_penalty
    lo, hi = config.length_ideal_min, config.length_ideal_max
    strength = config.length_penalty_strength
    def reward_length_penalty(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            length = len(text)
            if length < lo:
                penalty = (lo - length) * strength
            elif length > hi:
                penalty = (length - hi) * strength
            else:
                penalty = 0.0
            rewards.append(-w * penalty)   # 负号：惩罚
        return rewards
    reward_length_penalty.__name__ = "reward_length_penalty"
    return reward_length_penalty


def build_reward_funcs(config: RewardConfig):
    """
    根据 RewardConfig 构建奖励函数列表（权重>0 的分量才纳入）。
    返回 (reward_funcs, active_names) 供 GRPOTrainer 使用。
    """
    funcs = []
    if config.weight_correct != 0:
        funcs.append(make_reward_correct(config))
    if config.weight_format != 0:
        funcs.append(make_reward_format(config))
    if config.weight_cot_step != 0:
        funcs.append(make_reward_cot_step(config))
    if config.weight_length_penalty != 0:
        funcs.append(make_reward_length_penalty(config))
    if not funcs:
        raise ValueError("至少需要一个非零权重的奖励分量")
    return funcs, [f.__name__ for f in funcs]


def compute_total_reward(text: str, answer: int, config: RewardConfig) -> dict:
    """
    纯函数版：给定单条输出和配置，计算各分量与总分。
    供单测和离线分析使用（不依赖 TRL）。

    返回: {"correct": float, "format": float, "cot_step": float,
           "length_penalty": float, "total": float}
    """
    fmt_ok, _, loose_ok = parse_output(text, answer)
    parts = {
        "correct": config.weight_correct if loose_ok else 0.0,
        "format": config.weight_format if fmt_ok else 0.0,
        "cot_step": config.weight_cot_step if any(p.search(text) for p in COT_PATTERNS) else 0.0,
        "length_penalty": 0.0,
    }
    length = len(text)
    if config.weight_length_penalty != 0:
        if length < config.length_ideal_min:
            parts["length_penalty"] = -config.weight_length_penalty * (config.length_ideal_min - length) * config.length_penalty_strength
        elif length > config.length_ideal_max:
            parts["length_penalty"] = -config.weight_length_penalty * (length - config.length_ideal_max) * config.length_penalty_strength
    parts["total"] = sum(parts.values())
    return parts


if __name__ == "__main__":
    # 快速自检
    print("=== 奖励配置自检 ===")
    cfg = RewardConfig(weight_cot_step=0.3, weight_length_penalty=1.0)
    tests = [
        ("<answer>85</answer>", 85, "对+格式"),
        ("85", 85, "对无格式"),
        ("<answer>90</answer>", 85, "格式但错"),
        ("首先 47+38=85，因此 <answer>85</answer>", 85, "对+格式+CoT"),
        ("<answer>85</answer>" + " " * 100, 85, "对但过长"),
    ]
    print(f"配置: correct={cfg.weight_correct}, format={cfg.weight_format}, "
          f"cot={cfg.weight_cot_step}, len_pen={cfg.weight_length_penalty}\n")
    for text, ans, desc in tests:
        r = compute_total_reward(text, ans, cfg)
        print(f"[{desc}]")
        print(f"  输出: {text[:40]}{'...' if len(text)>40 else ''}")
        print(f"  分量: {r}")
