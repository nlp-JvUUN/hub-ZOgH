"""
难度题库单测：纯 CPU 运行，无需 GPU/模型/trl

覆盖：
  1. 所有难度（含新增 L7/L8/L9）能正常生成题目
  2. 生成的答案与表达式一致（用 eval 验证，带 ×÷ 转换）
  3. 课程配比校验逻辑（合法/非法输入）
  4. 课程生成的题数分布与配比一致
  5. parse_output 对新难度的解析正确

运行：
  python src/test_levels.py
  python -m pytest src/test_levels.py -v  （如装了 pytest）
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arithmetic_levels import (
    LEVELS, LEVEL_DESC, LEVEL_MIX_DEFAULT, LEVEL_MIX_EXTENDED,
    make_problem, parse_output, validate_level_mix, build_curriculum,
)


def _safe_eval(expr: str) -> int:
    """安全求值算术表达式（仅含数字和 + - × ÷ 及括号）。"""
    py_expr = expr.replace("×", "*").replace("÷", "//")
    return int(eval(py_expr, {"__builtins__": {}}, {}))


def test_all_levels_generate():
    """每个难度都能生成 100 道不报错的题。"""
    rng = random.Random(0)
    for lv in LEVELS:
        for _ in range(100):
            expr, ans = make_problem(lv, rng)
            assert isinstance(expr, str) and expr
            assert isinstance(ans, int)
    print("[PASS] 所有 9 个难度各生成 100 题无异常")


def test_answers_match_expressions():
    """生成的标准答案与表达式求值结果一致。"""
    rng = random.Random(42)
    for lv in LEVELS:
        for _ in range(200):
            expr, ans = make_problem(lv, rng)
            computed = _safe_eval(expr)
            assert computed == ans, f"{lv}: '{expr}' 求值={computed} 但 answer={ans}"
    print("[PASS] 所有难度的表达式求值与标准答案一致（每级 200 题）")


def test_non_negative_answers():
    """所有难度的答案都非负（减法已保证大减小）。"""
    rng = random.Random(7)
    for lv in LEVELS:
        for _ in range(100):
            _, ans = make_problem(lv, rng)
            assert ans >= 0, f"{lv} 出现负答案: {ans}"
    print("[PASS] 所有难度答案非负")


def test_parse_output_on_new_levels():
    """parse_output 对新难度格式的输出能正确解析。"""
    cases = [
        ("<answer>60</answer>", 60, True, True, True),
        ("(12 + 8) × 3 = 60，<answer>60</answer>", 60, True, True, True),
        ("144 ÷ 12 = 12，<answer>12</answer>", 12, True, True, True),
        ("60", 60, False, False, True),
        ("<answer>59</answer>", 60, True, False, False),
        ("", 60, False, False, False),
    ]
    for text, answer, exp_fmt, exp_strict, exp_loose in cases:
        fmt, strict, loose = parse_output(text, answer)
        assert fmt == exp_fmt, f"格式判定错: '{text}' -> {fmt} != {exp_fmt}"
        assert strict == exp_strict, f"严格判定错: '{text}' -> {strict} != {exp_strict}"
        assert loose == exp_loose, f"宽松判定错: '{text}' -> {loose} != {exp_loose}"
    print("[PASS] parse_output 对新难度输出解析正确")


def test_validate_level_mix_legal():
    """合法配比通过校验。"""
    validate_level_mix(LEVEL_MIX_DEFAULT)
    validate_level_mix(LEVEL_MIX_EXTENDED)
    validate_level_mix({"L1_add_1digit": 1.0})
    print("[PASS] 合法配比通过校验")


def test_validate_level_mix_illegal():
    """非法配比被拒绝。"""
    bad_cases = [
        {},
        {"L99": 1.0},
        {"L1_add_1digit": -0.5, "L2_addsub_2digit": 1.5},
        {"L1_add_1digit": 0.5, "L2_addsub_2digit": 0.4},
    ]
    for bad in bad_cases:
        try:
            validate_level_mix(bad)
            raise AssertionError(f"应拒绝非法配比: {bad}")
        except ValueError:
            pass
    print("[PASS] 非法配比被正确拒绝")


def test_curriculum_distribution():
    """课程生成的题数分布与配比一致（容差 ±2%）。"""
    n = 1000
    rows = build_curriculum(LEVEL_MIX_EXTENDED, n=n, seed=42)
    assert len(rows) == n
    from collections import Counter
    dist = Counter(r["level"] for r in rows)
    for lv, p in LEVEL_MIX_EXTENDED.items():
        actual = dist[lv] / n
        assert abs(actual - p) <= 0.02, f"{lv} 实际 {actual:.3f} vs 目标 {p:.3f}"
    print(f"[PASS] 课程配比分布正确（n={n}，容差 2%）")


def test_curriculum_shuffled():
    """课程生成后题目是打乱的（不按难度连续排列）。"""
    rows = build_curriculum(LEVEL_MIX_DEFAULT, n=100, seed=42)
    levels_seq = [r["level"] for r in rows]
    # 若未打乱，前 50 题应全是同一难度
    assert len(set(levels_seq[:10])) > 1, "题目似乎未被打乱"
    print("[PASS] 课程生成后题目已打乱")


def main():
    print("=" * 60)
    print("难度题库单测（纯 CPU，无需 GPU/模型）")
    print("=" * 60)
    test_all_levels_generate()
    test_answers_match_expressions()
    test_non_negative_answers()
    test_parse_output_on_new_levels()
    test_validate_level_mix_legal()
    test_validate_level_mix_illegal()
    test_curriculum_distribution()
    test_curriculum_shuffled()
    print("=" * 60)
    print("全部通过 [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
