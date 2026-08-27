"""使用 TRL 的 GRPOTrainer 对小型指令模型进行算术强化学习。"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset

import trl_compat  # noqa: F401，兼容特定 TRL/Transformers 组合
from trl import GRPOConfig, GRPOTrainer

from math_utils import completion_text, make_problem, messages_for, parse_answer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_LEVELS = [
    ("L3_addsub_3digit", 0.50),
    ("L5_mul_2x1digit", 0.25),
    ("L2_addsub_2digit", 0.25),
]


def choose_level(rng: random.Random) -> str:
    value, cumulative = rng.random(), 0.0
    for level, probability in TRAIN_LEVELS:
        cumulative += probability
        if value <= cumulative:
            return level
    return TRAIN_LEVELS[-1][0]


def build_dataset(size: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        level = choose_level(rng)
        expression, answer = make_problem(level, rng)
        rows.append({
            "prompt": messages_for(expression),
            "answer": answer,
            "level": level,
        })
    return Dataset.from_list(rows)


def reward_correct(completions, answer, **kwargs):
    """答案正确奖励。宽松解析避免模型未学会标签时奖励全部为零。"""
    return [
        1.0 if parse_answer(completion_text(c), int(a))[2] else 0.0
        for c, a in zip(completions, answer)
    ]


def reward_format(completions, **kwargs):
    """输出遵循 <answer>...</answer> 格式时给予小额奖励。"""
    return [
        0.2 if parse_answer(completion_text(c), 0)[0] else 0.0
        for c in completions
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 数学能力训练")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 模型名或本地路径")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true", help="使用 LoRA，降低显存需求")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "grpo_model"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("本实验训练需要 CUDA GPU；无 GPU 时可阅读已附实验结果。")
    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch-size 必须能被 num-generations 整除。")

    output_dir = Path(args.output_dir)
    dataset = build_dataset(args.dataset_size, args.seed)
    peft_config = None
    learning_rate = args.learning_rate
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        learning_rate = 2e-4

    config = GRPOConfig(
        output_dir=str(output_dir),
        model_init_kwargs={"torch_dtype": "bfloat16"},
        num_generations=args.num_generations,
        max_prompt_length=128,
        max_completion_length=64,
        temperature=1.0,
        beta=0.0,
        epsilon=0.2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=learning_rate,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=[reward_correct, reward_format],
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(str(output_dir))
    with (output_dir.parent / "train_log_new.json").open("w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)
    print(f"训练完成，模型保存在：{output_dir}")


if __name__ == "__main__":
    main()
