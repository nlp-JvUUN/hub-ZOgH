"""数据格式：加载 GSM8K / MATH，统一为 {problem, answer}，并构造 chat prompt。"""
import re
from typing import List, Dict

from datasets import load_dataset

# 要求模型把最终答案放进 \\boxed{} 中，便于奖励函数解析
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}.\n\n"
    "Problem: {problem}\n\n"
    "Solution: "
)


def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K 的 answer 形如 '... #### 42'，取 #### 后的数字。"""
    match = re.search(r"####\s*(.+)", answer_text.strip())
    return match.group(1).strip() if match else answer_text.strip()


def extract_math_answer(solution_text: str) -> str:
    """MATH 数据集的 solution 内部用 \\boxed{...} 标注答案。"""
    ans = _extract_last_boxed(solution_text)
    return ans if ans is not None else ""


def _extract_last_boxed(text: str):
    """返回最后一个 \\boxed{...} 的内部内容（支持嵌套花括号）。"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            return text[start:i]
        i += 1
    return text[start:]


def load_math_dataset(name: str = "gsm8k", split: str = "train",
                      n_samples: int = -1) -> List[Dict[str, str]]:
    """加载数据集，返回 [{"problem":..., "answer":...}, ...]。"""
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        examples = [
            {"problem": ex["question"],
             "answer": extract_gsm8k_answer(ex["answer"])}
            for ex in ds
        ]
    elif name == "math":
        # MATH 是评测集，通常只有 test/validation split（无 train）；
        # 不同镜像 repo 名不一，逐一尝试以提高可用性。
        last_err = None
        ds = None
        for repo in ("HuggingFaceH4/MATH", "lighteval/MATH"):
            try:
                ds = load_dataset(repo, "all", split=split)
                break
            except Exception as e:  # 数据集不存在或 split 缺失
                last_err = e
        if ds is None:
            raise RuntimeError(
                f"无法加载 MATH 数据集（split={split}），请改用 --dataset_split test：{last_err}"
            )
        examples = [
            {"problem": ex["problem"],
             "answer": extract_math_answer(ex["solution"])}
            for ex in ds
        ]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if n_samples is not None and n_samples > 0:
        examples = examples[:n_samples]
    return examples


def build_prompt(problem: str, tokenizer) -> str:
    """用 chat template 构造提示文本（带 generation prompt）。"""
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text
