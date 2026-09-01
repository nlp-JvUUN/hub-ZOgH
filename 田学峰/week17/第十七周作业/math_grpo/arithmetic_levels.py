"""
难度题库扩展：在原 6 级难度基础上新增 L7/L8，并支持自定义难度与课程配比

设计原则：
  1. 与原 probe_baseline.py 的 make_problem / parse_output 完全兼容，原 6 级难度行为不变
  2. 新增难度遵循同样的契约：make_problem(level, rng) -> (expr_text, answer_int)
  3. LEVELS 列表是"全集"，课程训练可从中任意子集+配比
  4. parse_output 保持原逻辑（宽松/严格双口径），新增难度无需改解析

新增难度：
  L7_paren_arith   带括号的四则运算，如 (12 + 8) × 3 = 60
  L8_division      整除除法，如 144 ÷ 12 = 12（保证整除，无余数）
  L9_mixed_ops     混合四则运算，如 12 + 8 × 3 = 36（考察运算优先级）

使用方式：
  from arithmetic_levels import LEVELS, make_problem, LEVEL_MIX_DEFAULT, build_curriculum
  problems = build_curriculum([("L3_addsub_3digit", 0.5), ("L5_mul_2x1digit", 0.5)], n=100, seed=42)

教学点：
  - 新增难度后应先跑 probe_baseline 摸 informative group rate，再决定是否进训练集
  - L7/L8 是验证"算术能力泛化"的好素材：和 L3/L5 共享数字处理回路，但题型不同
  - L9 考察运算优先级，是向"真实数学题"过渡的一步
"""
import random
import re
from typing import List, Tuple

# ── 复用原脚本的解析逻辑，保持一致性 ──────────────────────────────────────
TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")


def parse_output(text: str, answer: int):
    """与 probe_baseline.parse_output 完全一致的解析（导入兼容用）。"""
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer
    return fmt_ok, strict_ok, loose_ok


# ── 难度全集（原 6 级 + 新增 3 级）─────────────────────────────────────────
LEVELS: List[str] = [
    "L1_add_1digit",        # 个位数加法
    "L2_addsub_2digit",     # 两位数加减
    "L3_addsub_3digit",     # 三位数加减
    "L4_mul_1digit",        # 表内乘法
    "L5_mul_2x1digit",      # 两位×一位
    "L6_mul_2x2digit",      # 两位×两位
    "L7_paren_arith",       # 新增：带括号四则运算
    "L8_division",          # 新增：整除除法
    "L9_mixed_ops",         # 新增：混合四则运算（考优先级）
]

# 各难度的简短描述，供 probe 表格展示
LEVEL_DESC = {
    "L1_add_1digit": "个位数加法",
    "L2_addsub_2digit": "两位数加减",
    "L3_addsub_3digit": "三位数加减",
    "L4_mul_1digit": "表内乘法",
    "L5_mul_2x1digit": "两位×一位",
    "L6_mul_2x2digit": "两位×两位",
    "L7_paren_arith": "带括号四则运算",
    "L8_division": "整除除法",
    "L9_mixed_ops": "混合四则运算",
}


def make_problem(level: str, rng: random.Random) -> Tuple[str, int]:
    """
    按难度级别生成一道算术题，返回 (表达式文本, 标准答案)。

    原 6 级逻辑与 probe_baseline.make_problem 一致；新增 L7/L8/L9。
    """
    # ── 原 6 级（与 probe_baseline 完全一致，保证向后兼容）──────────────
    if level == "L1_add_1digit":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"{a} + {b}", a + b
    if level == "L2_addsub_2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)
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

    # ── 新增难度 ────────────────────────────────────────────────────────
    if level == "L7_paren_arith":
        # 带括号的四则运算：(a op b) op c，答案控制在两位数以内便于 0.5B 学习
        # 难度旋钮：括号内两位数加减，括号外乘一位数
        a = rng.randint(10, 49)
        b = rng.randint(1, 20)
        c = rng.randint(2, 9)
        if rng.random() < 0.5:
            inner, inner_ans = f"{a} + {b}", a + b
        else:
            inner_a, inner_b = max(a, b), min(a, b)
            inner, inner_ans = f"{inner_a} - {inner_b}", inner_a - inner_b
        # 一半乘一半加，保证多样性
        if rng.random() < 0.5:
            return f"({inner}) × {c}", inner_ans * c
        return f"({inner}) + {c}", inner_ans + c

    if level == "L8_division":
        # 整除除法：先生成除数和商，再反推被除数，保证无余数
        divisor = rng.randint(2, 12)
        quotient = rng.randint(2, 15)
        dividend = divisor * quotient
        return f"{dividend} ÷ {divisor}", quotient

    if level == "L9_mixed_ops":
        # 混合四则运算：a + b × c 或 a × b - c，考运算优先级
        a = rng.randint(2, 20)
        b = rng.randint(2, 9)
        c = rng.randint(1, 20)
        pattern = rng.randint(0, 3)
        if pattern == 0:   # a + b × c  （先乘后加）
            return f"{a} + {b} × {c}", a + b * c
        if pattern == 1:   # a × b - c  （先乘后减）
            ans = a * b - c
            if ans < 0:    # 保证非负
                a, c = c, a
                ans = a * b - c
            return f"{a} × {b} - {c}", ans
        if pattern == 2:   # a × b + c
            return f"{a} × {b} + {c}", a * b + c
        # a - b × c（先乘后减）
        ans = a - b * c
        if ans < 0:
            a = b * c + rng.randint(1, 10)
            ans = a - b * c
        return f"{a} - {b} × {c}", ans

    raise ValueError(f"未知难度级别: {level}")


# ── 课程配置：默认配比（与原 train_grpo.LEVEL_MIX 一致）───────────────────
# 键为难度名，值为占比（0~1，总和应=1）。训练脚本会校验。
LEVEL_MIX_DEFAULT = {
    "L3_addsub_3digit": 0.50,
    "L5_mul_2x1digit": 0.25,
    "L2_addsub_2digit": 0.25,
}

# 扩展课程示例：加入新难度，验证泛化到新题型
LEVEL_MIX_EXTENDED = {
    "L3_addsub_3digit": 0.30,
    "L5_mul_2x1digit": 0.25,
    "L7_paren_arith": 0.20,
    "L8_division": 0.15,
    "L2_addsub_2digit": 0.10,
}


def validate_level_mix(level_mix: dict) -> None:
    """校验课程配比：难度名合法、占比非负、总和≈1。"""
    if not level_mix:
        raise ValueError("level_mix 不能为空")
    unknown = set(level_mix) - set(LEVELS)
    if unknown:
        raise ValueError(f"未知难度名: {unknown}，合法难度: {LEVELS}")
    if any(p < 0 for p in level_mix.values()):
        raise ValueError(f"占比不能为负: {level_mix}")
    total = sum(level_mix.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"占比总和应为 1.0，当前 {total:.4f}: {level_mix}")


def build_curriculum(level_mix: dict, n: int, seed: int = 123) -> List[dict]:
    """
    按课程配比生成 n 道题，返回 rows 列表（可直接转 Dataset）。

    每行: {"expr": str, "answer": int, "level": str}
    """
    validate_level_mix(level_mix)
    rng = random.Random(seed)
    # 预计算每个难度的题数，避免逐题抽样带来的配比偏差
    counts = {lv: int(round(p * n)) for lv, p in level_mix.items()}
    # 修正取整误差：差额补到占比最大的难度
    diff = n - sum(counts.values())
    if diff != 0:
        max_lv = max(level_mix, key=level_mix.get)
        counts[max_lv] += diff

    rows = []
    for lv, cnt in counts.items():
        for _ in range(cnt):
            expr, ans = make_problem(lv, rng)
            rows.append({"expr": expr, "answer": ans, "level": lv})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    # 快速自检：每个难度生成 3 题并打印
    print("=== 难度题库自检 ===")
    rng = random.Random(0)
    for lv in LEVELS:
        print(f"\n[{lv}] {LEVEL_DESC[lv]}")
        for _ in range(3):
            expr, ans = make_problem(lv, rng)
            print(f"  {expr} = {ans}")

    print("\n=== 课程生成自检（扩展配比，n=20）===")
    rows = build_curriculum(LEVEL_MIX_EXTENDED, n=20, seed=42)
    from collections import Counter
    dist = Counter(r["level"] for r in rows)
    for lv in LEVEL_MIX_EXTENDED:
        print(f"  {lv}: {dist[lv]} 题 ({dist[lv]/20:.0%}, 目标 {LEVEL_MIX_EXTENDED[lv]:.0%})")
    print(f"  总计: {len(rows)} 题")
