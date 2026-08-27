"""算术题生成、提示词构造与答案解析工具。"""
import random
import re

SYSTEM_PROMPT = (
    "你是一个数学助手。请准确计算题目，并且只输出最终答案，"
    "格式必须为 <answer>整数</answer>。"
)

LEVELS = [
    "L1_add_1digit",
    "L2_addsub_2digit",
    "L3_addsub_3digit",
    "L4_mul_1digit",
    "L5_mul_2x1digit",
    "L6_mul_2x2digit",
]

ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+")


def make_problem(level: str, rng: random.Random):
    """按难度生成 (表达式, 标准答案)。"""
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
    raise ValueError(f"未知难度：{level}")


def messages_for(expr: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"计算：{expr} = ?"},
    ]


def completion_text(completion) -> str:
    """兼容 TRL 的聊天补全和纯字符串补全。"""
    if isinstance(completion, str):
        return completion
    if completion and isinstance(completion, list):
        item = completion[0]
        return item.get("content", "") if isinstance(item, dict) else str(item)
    return str(completion)


def parse_answer(text: str, expected: int):
    """返回 (格式正确, 严格正确, 宽松正确)。"""
    match = ANSWER_RE.search(text)
    format_ok = match is not None
    strict_ok = format_ok and int(match.group(1)) == int(expected)
    numbers = NUMBER_RE.findall(text)
    loose_ok = bool(numbers) and int(numbers[-1]) == int(expected)
    return format_ok, strict_ok, loose_ok
