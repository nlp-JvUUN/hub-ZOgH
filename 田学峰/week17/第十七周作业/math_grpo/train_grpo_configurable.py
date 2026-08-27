"""
可配置 GRPO 训练：在原 train_grpo.py 基础上全参数化，支持任意模型/难度配比/奖励权重

与原 train_grpo.py 的区别：
  1. --model 可指定任意基座模型路径（不再硬编码 Qwen2-0.5B）
  2. --level-mix 可指定训练难度配比（支持新增的 L7/L8/L9）
  3. --reward-* 可调各奖励分量权重（正确/格式/CoT/长度惩罚）
  4. --num-generations / --beta / --epsilon / --temperature 等核心超参全部可配
  5. 自动复用 arithmetic_levels + rewards 模块，保持与原版一致的奖励解析口径

使用方式：
  # 默认配置（与原 train_grpo.py 等价）
  python src/train_grpo_configurable.py

  # 换模型 + 扩展难度 + 开 CoT 奖励
  python src/train_grpo_configurable.py \
      --model /path/to/Qwen2.5-1.5B-Instruct \
      --level-mix L3_addsub_3digit:0.3,L5_mul_2x1digit:0.25,L7_paren_arith:0.2,L8_division:0.15,L2_addsub_2digit:0.1 \
      --reward-cot-step 0.3 --max-completion-length 128 --tag ext

  # LoRA + 调高温度保持多样性
  python src/train_grpo_configurable.py --lora --temperature 1.2 --tag hot

注意：
  - 本脚本依赖 trl，需在有 GPU + trl 的环境运行（本机 CPU 环境仅用于单测）
  - 必须从项目根目录运行，确保 import trl_compat / arithmetic_levels / rewards 可达
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset

import trl_compat  # noqa: F401  必须先于 trl 导入
from trl import GRPOConfig, GRPOTrainer

from arithmetic_levels import (
    LEVELS, LEVEL_MIX_DEFAULT, LEVEL_MIX_EXTENDED,
    validate_level_mix, make_problem, LEVEL_DESC,
)
from rewards import RewardConfig, build_reward_funcs
from probe_baseline import SYSTEM_PROMPT

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs"
# 默认模型路径与原版一致，可通过 --model 覆盖
DEFAULT_MODEL = r"D:\badou\八斗课程\pretrain_models\Qwen2-0.5B-Instruct"


def parse_level_mix(s: str) -> dict:
    """解析 'L3_addsub_3digit:0.5,L5_mul_2x1digit:0.5' 格式为 dict。"""
    if not s:
        return dict(LEVEL_MIX_DEFAULT)
    if s == "extended":
        return dict(LEVEL_MIX_EXTENDED)
    result = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"难度配比格式应为 'L3_addsub_3digit:0.5'，收到 '{item}'")
        lv, p = item.rsplit(":", 1)
        try:
            result[lv] = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"占比应为数字: '{p}'")
    return result


def build_dataset(level_mix: dict, n: int, seed: int) -> Dataset:
    """按课程配比生成训练集，格式与原 train_grpo.build_dataset 一致。"""
    validate_level_mix(level_mix)
    import random
    rng = random.Random(seed)
    # 预计算每难度题数
    counts = {lv: int(round(p * n)) for lv, p in level_mix.items()}
    diff = n - sum(counts.values())
    if diff != 0:
        max_lv = max(level_mix, key=level_mix.get)
        counts[max_lv] += diff

    rows = []
    for lv, cnt in counts.items():
        for _ in range(cnt):
            expr, ans = make_problem(lv, rng)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"计算：{expr} = ?"},
                ],
                "answer": ans,
                "level": lv,
            })
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="可配置 GRPO 训练（扩展版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── 模型 ──────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="基座模型路径")
    # ── 难度课程 ──────────────────────────────────────────────────────────
    parser.add_argument("--level-mix", type=str, default="",
                        help="训练难度配比，格式 'L3:0.5,L5:0.5'；留空=默认配比；'extended'=扩展配比")
    parser.add_argument("--n-prompts", type=int, default=1000, help="训练集 prompt 数")
    # ── 奖励权重 ──────────────────────────────────────────────────────────
    parser.add_argument("--reward-correct", type=float, default=1.0, help="正确分权重")
    parser.add_argument("--reward-format", type=float, default=0.2, help="格式分权重")
    parser.add_argument("--reward-cot-step", type=float, default=0.0,
                        help="CoT 步骤奖励权重（开此需配合 --max-completion-length 加大）")
    parser.add_argument("--reward-length-penalty", type=float, default=0.0,
                        help="长度惩罚权重")
    # ── GRPO 核心超参 ─────────────────────────────────────────────────────
    parser.add_argument("--num-generations", type=int, default=8, help="组内采样数 K")
    parser.add_argument("--beta", type=float, default=0.0, help="KL 系数（0=不加载参考模型）")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO-clip 裁剪范围")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=64)
    # ── 训练超参 ──────────────────────────────────────────────────────────
    parser.add_argument("--lr", type=float, default=2e-6, help="全量微调学习率")
    parser.add_argument("--max-steps", type=int, default=200, help="优化步数")
    parser.add_argument("--per-device-batch", type=int, default=8, help="每设备微批次")
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lora", action="store_true", help="降级为 LoRA")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA 目标模块（逗号分隔）")
    # ── 输出 ──────────────────────────────────────────────────────────────
    parser.add_argument("--tag", type=str, default="", help="输出目录后缀")
    parser.add_argument("--log-completions", action="store_true", help="打印每步采样")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")
    args = parser.parse_args()

    # ── 解析并校验配置 ────────────────────────────────────────────────────
    level_mix = parse_level_mix(args.level_mix)
    print(f"训练难度配比: {level_mix}")

    reward_config = RewardConfig(
        weight_correct=args.reward_correct,
        weight_format=args.reward_format,
        weight_cot_step=args.reward_cot_step,
        weight_length_penalty=args.reward_length_penalty,
    )
    reward_funcs, active_names = build_reward_funcs(reward_config)
    print(f"启用奖励分量: {active_names}")

    suffix = f"_{args.tag}" if args.tag else ""
    mode = "lora" if args.lora else "full"
    ckpt_dir = OUT_DIR / f"grpo_{mode}_ckpt{suffix}"
    log_path = OUT_DIR / f"train_log_{mode}{suffix}.json"

    # ── 构建数据集 ────────────────────────────────────────────────────────
    dataset = build_dataset(level_mix, args.n_prompts, seed=123)
    print(f"训练集大小: {len(dataset)}")

    # ── LoRA 配置 ─────────────────────────────────────────────────────────
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_targets.split(","),
        )

    # ── GRPO 配置 ─────────────────────────────────────────────────────────
    config = GRPOConfig(
        output_dir=str(ckpt_dir),
        model_init_kwargs={"torch_dtype": "bfloat16"},
        num_generations=args.num_generations,
        beta=args.beta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr if not args.lora else 2e-4,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=False,  # transformers 5.x 下会损坏 generate，必须关
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        log_completions=args.log_completions,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    trainer.processing_class.save_pretrained(str(ckpt_dir))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # 保存本次实验配置，便于复现
    config_path = OUT_DIR / f"train_config_{mode}{suffix}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "level_mix": level_mix,
            "reward_config": reward_config.__dict__,
            "active_rewards": active_names,
            "grpo": {
                "num_generations": args.num_generations,
                "beta": args.beta,
                "epsilon": args.epsilon,
                "temperature": args.temperature,
                "max_completion_length": args.max_completion_length,
            },
            "training": {
                "lr": args.lr if not args.lora else 2e-4,
                "max_steps": args.max_steps,
                "lora": args.lora,
                "seed": args.seed,
            },
        }, f, ensure_ascii=False, indent=2)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n训练完成。checkpoint: {ckpt_dir}")
    print(f"训练日志: {log_path}")
    print(f"实验配置: {config_path}")
    print(f"GPU 峰值显存: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
