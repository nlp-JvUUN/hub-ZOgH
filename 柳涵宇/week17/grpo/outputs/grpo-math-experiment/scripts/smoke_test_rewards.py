from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grpo_math.rewards import boxed_format_reward, extract_answer, math_accuracy_reward


def main() -> None:
    completions = [
        "We compute 2 + 2 = 4, so the final answer is \\boxed{4}.",
        "The answer is 5.",
        [{"role": "assistant", "content": "Therefore \\boxed{7/2}."}],
    ]
    solutions = ["#### 4", "#### 4", "\\boxed{3.5}"]

    print("Extracted:", [extract_answer(str(item)) for item in completions])
    print("Accuracy:", math_accuracy_reward(completions, solution=solutions))
    print("Boxed:", boxed_format_reward(completions))


if __name__ == "__main__":
    main()
