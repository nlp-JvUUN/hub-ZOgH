"""无需模型和 GPU 的基础单元测试：python src/test_math_utils.py"""
import random

from math_utils import LEVELS, make_problem, parse_answer


def main():
    rng = random.Random(42)
    for level in LEVELS:
        expression, answer = make_problem(level, rng)
        assert isinstance(expression, str) and isinstance(answer, int)

    assert parse_answer("<answer>42</answer>", 42) == (True, True, True)
    assert parse_answer("42", 42) == (False, False, True)
    assert parse_answer("过程 40，最终 42", 42) == (False, False, True)
    assert parse_answer("<answer>-7</answer>", -7) == (True, True, True)
    assert parse_answer("<answer>41</answer>", 42) == (True, False, False)
    assert parse_answer("", 42) == (False, False, False)
    print("math_utils 测试全部通过。")


if __name__ == "__main__":
    main()
