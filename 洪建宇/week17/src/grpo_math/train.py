"""GRPO 数学题训练入口。

示例：
  python -m grpo_math.train \
      --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
      --dataset_name gsm8k --num_train_samples 200 \
      --num_generations 8 --prompt_batch_size 4 \
      --learning_rate 1e-6 --num_epochs 1
"""
import argparse

from .config import GRPOConfig
from .data import load_math_dataset
from .grpo_trainer import GRPOTrainer


def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser(description="GRPO 数学题强化学习训练")
    # 模型
    p.add_argument("--model_name_or_path", type=str, default=GRPOConfig().model_name_or_path)
    p.add_argument("--ref_model_name_or_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs")
    # 数据
    p.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_train_samples", type=int, default=1000)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    # GRPO
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.04)
    # 训练
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--prompt_batch_size", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # 奖励
    p.add_argument("--reward_correctness_weight", type=float, default=1.0)
    p.add_argument("--reward_format_weight", type=float, default=0.2)
    p.add_argument("--bf16", action="store_true", default=True)
    args = p.parse_args()
    return GRPOConfig(**vars(args))


def main():
    config = parse_args()
    print("加载数据集 ...", config.dataset_name)
    examples = load_math_dataset(config.dataset_name, config.dataset_split,
                                 config.num_train_samples)
    print(f"  共 {len(examples)} 条样本")

    trainer = GRPOTrainer(config)
    trainer.train(examples)


if __name__ == "__main__":
    main()
