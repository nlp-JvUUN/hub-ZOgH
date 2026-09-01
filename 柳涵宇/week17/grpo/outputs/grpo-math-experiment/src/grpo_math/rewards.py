"""Reward helpers for math-focused GRPO training."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def completion_to_text(completion: Any) -> str:
    """Normalize TRL completion payloads into plain text."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(completion, dict):
        return str(completion.get("content", completion))

    return str(completion)


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"^\\boxed\{(.+)\}$", r"\1", answer)
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.strip().rstrip(".")
    return answer


def extract_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return normalize_answer(boxed)

    if "####" in text:
        return normalize_answer(text.split("####")[-1])

    return normalize_answer(extract_last_number(text))


def _as_fraction(answer: str) -> Fraction | None:
    answer = normalize_answer(answer)
    if not answer:
        return None

    try:
        return Fraction(answer)
    except Exception:
        pass

    try:
        return Fraction(float(answer)).limit_denominator(1_000_000)
    except Exception:
        return None


def answers_equal(prediction: str, target: str) -> bool:
    prediction = normalize_answer(prediction)
    target = normalize_answer(target)
    if not prediction or not target:
        return False

    try:
        from math_verify import parse, verify

        parsed_prediction = parse(prediction)
        parsed_target = parse(target)
        if parsed_prediction and parsed_target:
            return bool(verify(parsed_prediction, parsed_target))
    except Exception:
        pass

    prediction_fraction = _as_fraction(prediction)
    target_fraction = _as_fraction(target)
    if prediction_fraction is not None and target_fraction is not None:
        return prediction_fraction == target_fraction

    return prediction == target


def math_accuracy_reward(
    completions: list[Any],
    solution: list[str] | None = None,
    answer: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """Reward exact or mathematically equivalent final answers.

    TRL forwards dataset columns as keyword arguments. This function accepts
    either a `solution` column, as used by several math datasets, or an `answer`
    column, as used by GSM8K-style data.
    """
    targets = solution if solution is not None else answer
    if targets is None:
        return [0.0 for _ in completions]

    rewards: list[float] = []
    for completion, target in zip(completions, targets):
        text = completion_to_text(completion)
        pred = extract_answer(text)
        gold = extract_answer(str(target))
        rewards.append(1.0 if answers_equal(pred, gold) else 0.0)
    return rewards


def boxed_format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Small reward for putting the final answer in \\boxed{}."""
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(1.0 if extract_boxed_answer(text) is not None else 0.0)
    return rewards


def concise_reasoning_reward(completions: list[Any], **_: Any) -> list[float]:
    """Discourage empty, one-token, or heavily rambling generations."""
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        tokenish_len = len(text.split())
        if 8 <= tokenish_len <= 512:
            rewards.append(1.0)
        elif tokenish_len < 8:
            rewards.append(0.0)
        else:
            rewards.append(0.2)
    return rewards
