import os

from datasets import load_dataset

THINK_OPEN = "\x3cthink\x3e"
THINK_CLOSE = "\x3c/think\x3e"

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Solve the problem step by step.\n"
    f"Put your full reasoning inside {THINK_OPEN}...{THINK_CLOSE} tags.\n"
    f"After the {THINK_CLOSE} tag, put ONLY the final numeric answer inside \\boxed{{}} "
    "(e.g. \\boxed{72}). Do not include units inside \\boxed{}."
)


def extract_solution(answer: str) -> str:
    solution = answer.split("####")[-1].strip().replace(",", "")
    return solution


def to_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": extract_solution(example["answer"]),
    }


def main():
    os.makedirs("data", exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main")

    train = dataset["train"].map(to_example, remove_columns=["question", "answer"])
    test = dataset["test"].map(to_example, remove_columns=["question", "answer"])

    train = train.shuffle(seed=42)
    eval_set = test.select(range(128))

    train.to_parquet("data/train.parquet")
    eval_set.to_parquet("data/eval.parquet")
    test.to_parquet("data/test.parquet")

    print(f"train: {len(train)} samples -> data/train.parquet")
    print(f"eval:  {len(eval_set)} samples -> data/eval.parquet")
    print(f"test:  {len(test)} samples -> data/test.parquet")
    print("example:", train[0])


if __name__ == "__main__":
    main()
