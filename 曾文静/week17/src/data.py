# -*- coding: utf-8 -*-
"""
data.py —— 算术题程序化生成（难度分级，参考课件 grpo_arithmetic 的"难度旋钮"）
==============================================================================
奖励必须"可程序化验证"，所以题目由程序生成、答案由程序计算，评估零成本零噪声。
这正是 DeepSeek-R1 用 GRPO 训练数学推理的"可验证奖励（verifiable reward）"
思路的最小版本。

难度分级（全部由 probe 实测确定）：
  L1 两位数不进位加法  —— 易：SFT 后即接近满分（保底题，验证 RL 不损伤已有能力）
  L2 两位数进位加法    —— 主训练任务（GRPO 只训这一级；SFT 停在能力"相变"中段，
                            基线 ≈ 50-80%，给 GRPO 留出提升空间）
  L3 三位数不进位加法  —— 没训过的"泛化题"：测 RL 学到的列式加法算法能否迁移
  L4 两位数×两位数     —— 超出小模型能力边界（怎么训都 ≈0）：测 RL 能否凭空造能力
"""
import random

LEVELS = ["L1", "L2", "L3", "L4"]

# SFT 语料配比：L1 40% / L2 40% / L3 15% / L4 5%
# L4 只给 5%：让模型"会一点点乘法但很弱"（能力边界实验）
SFT_MIX = [("L1", 0.40), ("L2", 0.40), ("L3", 0.15), ("L4", 0.05)]

TAG_RATE = 0.2  # SFT 数据中带 <answer> 标签格式的比例（80% 裸答案；GRPO 只优化被奖励的行为）


def has_carry_add(a, b):
    while a > 0 or b > 0:
        if a % 10 + b % 10 >= 10:
            return True
        a //= 10
        b //= 10
    return False


def no_carry_add(a, b):
    return not has_carry_add(a, b)


def gen_problem(level, rng):
    """返回 (question, answer) —— question 是 'Q: 47+38=?' 形式。"""
    if level == "L1":                       # 两位数不进位加法
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        while not no_carry_add(a, b):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
    elif level == "L2":                     # 两位数进位加法
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        while not has_carry_add(a, b):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
    elif level == "L3":                     # 三位数不进位加法
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        while not no_carry_add(a, b):
            a, b = rng.randint(100, 999), rng.randint(100, 999)
    elif level == "L4":                     # 两位数×两位数（能力边界）
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        return f"Q: {a}×{b}=?", str(a * b)
    else:
        raise ValueError(level)
    return f"Q: {a}+{b}=?", str(a + b)


def prompt_of(question):
    """问题前缀（模型要续写的部分）：'Q: 47+38=?\nA: '"""
    return question + "\nA: "


def answer_of(question, answer, tagged=None):
    """答案后缀。tagged=None 时按全局 TAG_RATE 随机决定是否用 <answer> 标签。"""
    if tagged is None:
        tagged = random.random() < TAG_RATE
    return f"<answer>{answer}</answer>" if tagged else answer


def make_sft_dataset(n_total=6000, seed=0):
    """SFT 训练语料（固定 seed，可复现；供 train_sft.py 使用）。"""
    rng = random.Random(seed)
    data = []
    for _ in range(n_total):
        p = rng.random()
        acc = 0.0
        for lv, w in SFT_MIX:
            acc += w
            if p <= acc:
                break
        q, a = gen_problem(lv, rng)
        data.append((lv, q, a))
    return data


def make_eval_set(per_level=300, seed=42):
    """固定评估集（基线/训练后必须用同一份，seed 固定，前后可配对比较）。"""
    rng = random.Random(seed)
    out = {}
    for lv in LEVELS:
        out[lv] = [gen_problem(lv, rng) for _ in range(per_level)]
    return out


def full_text(question, answer, tagged=None):
    return prompt_of(question) + answer_of(question, answer, tagged)


if __name__ == "__main__":
    s = make_sft_dataset(10, seed=1)
    for lv, q, a in s:
        print(f"[{lv}] {full_text(q, a)}")
