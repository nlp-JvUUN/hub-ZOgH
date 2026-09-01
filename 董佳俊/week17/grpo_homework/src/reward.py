"""
reward.py — 题目生成、输出解析、复合奖励（本作业独立实现）

设计说明：
  GRPO 需要"可程序化验证"的奖励信号。本模块提供：
    1. 6 档难度算术题的程序化生成（make_problem），位数/运算即"难度旋钮"。
    2. 模型输出解析（parse_output），区分"是否符合 <answer> 格式"与"是否算对"。
    3. 两个解耦的奖励函数（正确性 1.0 / 格式 0.2），供 GRPO 训练循环使用。

奖励意义：
  - reward_correct（宽松解析）：取输出最后一个数字与标准答案比对。
    * 宽松解析保证冷启动阶段（模型还完全不会输出 <answer> 标签时）正确信号不为 0，
      从而让组内有梯度、训练能启动。
  - reward_format：输出包含 "<answer>数字</answer>" 即得分（与正确性解耦），
    权重 0.2 故意小于正确分 1.0，用于观察"主次信号竞争"的真实 RL 现象。
"""
from __future__ import annotations

import random
import re
from typing import List, Tuple

# 系统提示词：要求模型把最终答案放进 <answer> 标签
SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

# 解析用的正则
_TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
_NUM_RE = re.compile(r"-?\d+")

# 六个难度级别，从易到难（L1 最简单，L6 最难）
LEVELS: List[str] = [
    "L1_add_1digit",
    "L2_addsub_2digit",
    "L3_addsub_3digit",
    "L4_mul_1digit",
    "L5_mul_2x1digit",
    "L6_mul_2x2digit",
]

# 训练集难度配比（依据 informative group rate 的"可学习性"选题）：
#   L3 / L5 为主（组内有对有错比例高，GRPO 梯度充分），L2 保底。
#   L1 / L4 / L6 不进训练集，留作未训练难度的泛化对照。
LEVEL_MIX: List[Tuple[str, float]] = [
    ("L3_addsub_3digit", 0.50),
    ("L5_mul_2x1digit", 0.25),
    ("L2_addsub_2digit", 0.25),
]


def make_problem(level: str, rng: random.Random) -> Tuple[str, int]:
    """按难度级别生成一道算术题，返回 (表达式文本, 标准答案)。"""
    if level == "L1_add_1digit":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"{a} + {b}", a + b
    if level == "L2_addsub_2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)  # 保证减法结果非负
        return f"{a} - {b}", a - b
    if level == "L3_addsub_3digit":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    if level == "L4_mul_1digit":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        return f"{a} × {b}", a * b
    if level == "L5_mul_2x1digit":
        a, b = rng.randint(10, 99), rng.randint(3, 9)
        return f"{a} × {b}", a * b
    if level == "L6_mul_2x2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a} × {b}", a * b
    raise ValueError(f"unknown level: {level}")


def sample_problem(rng: random.Random, mix: List[Tuple[str, float]] = LEVEL_MIX) -> Tuple[str, int, str]:
    """按 LEVEL_MIX 的权重随机抽一个难度并生成题目。返回 (expr, ans, level)。"""
    r = rng.random()
    acc = 0.0
    level = mix[-1][0]
    for lv, p in mix:
        acc += p
        if r <= acc:
            level = lv
            break
    expr, ans = make_problem(level, rng)
    return expr, ans, level


def parse_output(text: str, answer: int) -> Tuple[bool, bool, bool]:
    """解析模型输出，返回 (是否符合格式, 严格正确, 宽松正确)。

    严格正确 = 标签内数字 == 标准答案。
    宽松正确 = 输出中最后一个数字 == 标准答案（冷启动时模型常无标签）。
    """
    m = _TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = _NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer
    return fmt_ok, strict_ok, loose_ok


def reward_correct(completions: List[str], answers: List[int]) -> List[float]:
    """正确分（宽松解析）：宽松正确记 1.0，否则 0.0。"""
    return [1.0 if parse_output(c, a)[2] else 0.0 for c, a in zip(completions, answers)]


def reward_format(completions: List[str]) -> List[float]:
    """格式分：输出包含 <answer>数字</answer> 记 0.2，否则 0.0。"""
    return [0.2 if parse_output(c, 0)[0] else 0.0 for c in completions]
