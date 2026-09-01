from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grpo_math.rewards import boxed_format_reward, concise_reasoning_reward, math_accuracy_reward


SYSTEM_PROMPT = (
    "You are a careful math solver. Show the reasoning briefly, then put only "
    "the final answer in \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a math model with GRPO.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset-name", default="trl-lib/DeepMath-103K")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-format", default=None, help="Optional explicit dataset format like json or csv.")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional local data files for dataset formats such as json or csv.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--solution-column", default="solution")
    parser.add_argument("--prompt-style", choices=["chat", "plain"], default="chat")
    parser.add_argument("--output-dir", default="outputs/grpo-math")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    return parser.parse_args()


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_plain_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def prepare_dataset(args: argparse.Namespace):
    load_kwargs: dict[str, Any] = {}
    if args.data_files:
        load_kwargs["data_files"] = args.data_files
    dataset_name = args.dataset_format or args.dataset_name
    dataset_config = args.dataset_config
    if dataset_name in {"json", "csv", "parquet", "text"}:
        dataset_config = None
    dataset = load_dataset(dataset_name, dataset_config, split=args.split, **load_kwargs)
    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    if args.prompt_column not in dataset.column_names:
        raise ValueError(f"Prompt column {args.prompt_column!r} not found: {dataset.column_names}")
    if args.solution_column not in dataset.column_names:
        raise ValueError(f"Solution column {args.solution_column!r} not found: {dataset.column_names}")

    def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row[args.prompt_column]
        if args.prompt_style == "plain":
            row["prompt"] = build_plain_prompt(str(prompt))
        elif isinstance(prompt, str):
            row["prompt"] = build_prompt(prompt)
        else:
            row["prompt"] = prompt
        row["solution"] = row[args.solution_column]
        return row

    return dataset.map(normalize_row)


def maybe_lora_config(args: argparse.Namespace):
    if not args.use_lora:
        return None

    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main() -> None:
    args = parse_args()
    train_dataset = prepare_dataset(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=args.bf16,
        report_to=args.report_to,
        reward_weights=[1.0, 0.1, 0.02],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[math_accuracy_reward, boxed_format_reward, concise_reasoning_reward],
        peft_config=maybe_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
