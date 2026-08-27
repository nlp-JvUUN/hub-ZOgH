"""奖励函数：从模型输出中解析答案并与 ground truth 比较。

提供两类可验证奖励：
- correctness_reward：答案是否正确（数值/字符串规范化后比较）
- format_reward：是否包含 \\boxed{} 结构
最终奖励按权重加权，并复用 data 模块里的嵌套花括号解析器。
"""
import re
from typing import Optional

from .data import _extract_last_boxed


def extract_model_answer(text: str) -> Optional[str]:
    """从模型输出中抽取最后一个 \\boxed{...} 的内容。"""
    return _extract_last_boxed(text)


def normalize_answer(answer: str) -> str:
    """答案归一化：去掉 $、空格、千分位逗号、等价 LaTeX 包裹。"""
    if answer is None:
        return ""
    a = answer.strip()
    # 去掉首尾的 $ 和空格
    a = a.strip().strip("$").strip()
    # 去掉千分位逗号 42,000 -> 42000
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(,\d*)?", a):
        a = a.replace(",", "")
    # 统一分数 \frac{a}{b} -> a/b
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        a = f"{m.group(1)}/{m.group(2)}"
    # 去掉多余空格
    a = a.replace(" ", "")
    return a


def _to_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def answer_equal(pred: str, gt: str) -> bool:
    """比较两个答案是否等价（字符串规范化 + 数值近似）。"""
    np, ng = normalize_answer(pred), normalize_answer(gt)
    if np == ng:
        return True
    fp, fg = _to_float(np), _to_float(ng)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-4 * max(1.0, abs(fg))
    return False


def correctness_reward(output: str, gt_answer: str) -> float:
    pred = extract_model_answer(output)
    if pred is None:
        return 0.0
    return 1.0 if answer_equal(pred, gt_answer) else 0.0


def format_reward(output: str) -> float:
    return 1.0 if "\\boxed{" in output else 0.0


def compute_reward(output: str, gt_answer: str,
                    correctness_weight: float = 1.0,
                    format_weight: float = 0.2):
    """加权总奖励。返回 (total, 各分量 dict)。"""
    c = correctness_reward(output, gt_answer)
    f = format_reward(output)
    total = correctness_weight * c + format_weight * f
    return total, {"correctness": c, "format": f}
