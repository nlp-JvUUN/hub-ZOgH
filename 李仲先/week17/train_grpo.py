import argparse
import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward

MODEL_PATH = "Qwen2.5-0.5B-Instruct"

BOXED_PATTERN = re.compile(r"\\boxed\{[^{}]*\}")


def boxed_format_reward(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [1.0 if BOXED_PATTERN.search(content) else 0.0 for content in contents]


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for math reasoning")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default="output/grpo-qwen-gsm8k")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_lora", action="store_true", help="full fine-tuning instead of LoRA")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(
        "parquet",
        data_files={"train": "data/train.parquet", "eval": "data/eval.parquet"},
    )
    if len(dataset["eval"]) > 32:
        dataset["eval"] = dataset["eval"].select(range(32))

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        eval_strategy="steps" if args.max_steps >= 25 else "no",
        eval_steps=25,
        per_device_eval_batch_size=2,
        num_generations_eval=2,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        seed=args.seed,
        log_completions=True,
        report_to=[],
        model_init_kwargs={"dtype": torch.bfloat16},
        reward_weights=[1.0, 0.2, 0.2],
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[accuracy_reward, think_format_reward, boxed_format_reward],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.processing_class.save_pretrained(args.output_dir)
    print(f"model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
