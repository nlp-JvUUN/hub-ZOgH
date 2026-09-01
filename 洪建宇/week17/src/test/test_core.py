"""核心逻辑单元测试，不下载任何模型，秒级完成。

运行：
    python tests/test_core.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # 仅 advantage 部分用到，环境已含

from grpo_math.data import (
    _extract_last_boxed, extract_gsm8k_answer, extract_math_answer,
)
from grpo_math.rewards import (
    normalize_answer, answer_equal, correctness_reward,
    format_reward, compute_reward,
)


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


results = []

# 1) 嵌套 \boxed{} 解析
results.append(check("extract_last_boxed 简单",
                     _extract_last_boxed(r"x=\boxed{42}") == "42"))
results.append(check("extract_last_boxed 嵌套",
                     _extract_last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"))
results.append(check("extract_last_boxed 取最后一个",
                     _extract_last_boxed(r"\boxed{1} and \boxed{2}") == "2"))
results.append(check("extract_last_boxed 无 boxed",
                     _extract_last_boxed("no box") is None))

# 2) GSM8K / MATH 标准答案抽取
results.append(check("gsm8k #### 抽取",
                     extract_gsm8k_answer("blah #### 7") == "7"))
results.append(check("math solution 抽取",
                     extract_math_answer(r"solve \boxed{\frac{3}{4}} done") == r"\frac{3}{4}"))

# 3) 答案归一化
results.append(check("normalize 去美元符", normalize_answer("$42$") == "42"))
results.append(check("normalize 千分位", normalize_answer("42,000") == "42000"))
results.append(check("normalize 分数", normalize_answer(r"\frac{1}{2}") == "1/2"))

# 4) 答案等价比较
results.append(check("equal 数值", answer_equal("42", "42")))
results.append(check("equal 数值近似", answer_equal("42.0", "42")))
results.append(check("equal 分数等价", answer_equal("1/2", r"\frac{1}{2}")))
results.append(check("not equal", not answer_equal("42", "43")))

# 5) 奖励函数
out = "step by step ... the answer is \\boxed{42}"
results.append(check("correctness 命中", correctness_reward(out, "42") == 1.0))
results.append(check("correctness 错答", correctness_reward(out, "43") == 0.0))
results.append(check("correctness 无 boxed", correctness_reward("no answer", "42") == 0.0))
results.append(check("format 有 boxed", format_reward(out) == 1.0))
results.append(check("format 无 boxed", format_reward("no box") == 0.0))

# 6) 加权总奖励
tot, parts = compute_reward(out, "42", 1.0, 0.2)
results.append(check("reward 加权", abs(tot - 1.2) < 1e-9))
results.append(check("reward 分量", parts == {"correctness": 1.0, "format": 1.0}))

# 7) 优势组内归一化（复刻 compute_advantages 逻辑做断言）
def advantages(rewards, G, eps=1e-8):
    r = rewards.view(-1, G)
    return ((r - r.mean(1, keepdim=True)) /
            (r.std(1, keepdim=True) + eps)).view(-1)

a = advantages(torch.tensor([0., 0., 1., 1., 0., 0., 0., 1.]), G=4)
results.append(check("advantage 组均值≈0",
                     abs(a.view(-1, 4).mean(1).mean().item()) < 1e-6))

print()
total = len(results)
passed = sum(results)
print(f"RESULT: {passed}/{total} " + ("ALL PASS" if passed == total else "SOME FAILED"))
sys.exit(0 if passed == total else 1)
