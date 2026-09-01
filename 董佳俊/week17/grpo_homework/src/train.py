"""
train.py — 主训练脚本（自实现 GRPO，CLI 控制）

使用方式：
  python src/train.py                    # 默认：12 步（约 5 组 prompt/步），CPU 上约 12-20 分钟
  python src/train.py --max_steps 2 --quick  # 冒烟：快速验证显存/流程
  python src/train.py --seed 42 --out outputs/train_log.json

训练循环：每个优化步 = micro_batch_size 个 prompt，每个 prompt 采样 num_generations 条，
组内归一化 advantage，PPO-clip surrogate，一次 AdamW 更新。

输出：
  outputs/ckpt/            最终 checkpoint（保存模型 + tokenizer）
  outputs/train_log.json   每步指标（reward, entropy, surrogate, loss, 退化组比例）
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_loop import grpo_step
from reward import LEVEL_MIX, SYSTEM_PROMPT, sample_problem

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
DEFAULT_MODEL = r"D:\八斗学习内容\pretrain_models\Qwen2-0.5B-Instruct"


def _build_chat_text(tokenizer, expr: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"计算：{expr} = ?"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _entropy_of_policy(model, tokenizer, texts, max_new=32):
    """用 greedy 前向粗估策略熵（调试信号）：对每个 prompt 的模型输出分布取平均熵。"""
    entropies = []
    for t in texts[:8]:
        enc = tokenizer(t, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)
        logp = torch.log_softmax(logits.float(), dim=-1)
        ent = -(probs * logp).sum(-1).mean().item()
        entropies.append(ent)
    return sum(entropies) / len(entropies) if entropies else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--micro_batch", type=int, default=2, help="每个优化步覆盖的 prompt 数")
    ap.add_argument("--num_generations", type=int, default=8, help="组内采样数 K")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4, help="每步对采样做几次梯度更新（GRPO 核心）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="train_log.json", help="训练日志路径（相对 outputs/ 或绝对路径）")
    ap.add_argument("--ckpt", default="ckpt", help="checkpoint 目录（相对 outputs/ 或绝对路径）")
    ap.add_argument("--quick", action="store_true", help="冒烟：每步只 1 个 prompt")
    args = ap.parse_args()

    def _resolve(p: str, default_name: str):
        """把相对路径规范化到 outputs/ 下；若已是绝对路径则直接用。"""
        p = Path(p)
        if p.is_absolute():
            return p
        # 若用户不小心传了带 outputs/ 前缀的相对路径，去掉以避免重复
        if str(p).startswith("outputs"):
            p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(default_name)
        return OUT / p

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (torch {torch.__version__})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    from reward import reward_correct, reward_format

    micro = 1 if args.quick else args.micro_batch
    log_history = []
    rng = random.Random(args.seed)

    for step in range(1, args.max_steps + 1):
        # ── 构建本轮 prompt（按 LEVEL_MIX 配比抽难度的题）────────────────────────
        batch = []
        for _ in range(micro):
            expr, ans, level = sample_problem(rng, LEVEL_MIX)
            batch.append((_build_chat_text(tokenizer, expr), ans, level))
        prompts = [b[0] for b in batch]
        answers = [b[1] for b in batch]

        stats = grpo_step(
            model, optimizer, prompts, answers, tokenizer,
            reward_correct, reward_format,
            num_generations=args.num_generations,
            epsilon=args.eps,
            epochs=args.epochs,
            log_completions=(step <= 2 or args.quick),
        )

        # 额外监控：策略熵
        entropy = _entropy_of_policy(model, tokenizer, prompts)
        log_history.append({
            "step": step,
            "loss": stats["pg_loss"],
            "surrogate": stats["surrogate"],
            "reward_mean": stats["reward_mean"],
            "entropy": entropy,
            "lr": args.lr,
        })
        print(f"step {step:>3}  loss={stats['pg_loss']:+.5f}  surrogate={stats['surrogate']:+.5f} "
              f"reward_mean={stats['reward_mean']:.4f}  entropy={entropy:.4f}")

    # ── 保存 checkpoint ─────────────────────────────────────────────────────
    ckpt = _resolve(args.ckpt, "ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    print(f"\ncheckpoint 已保存：{ckpt}")

    log_path = _resolve(args.out, "train_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)
    print(f"训练日志已保存：{log_path}")
    print(f"GPU/CPU 峰值显存：{torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0:.2f} GB")


if __name__ == "__main__":
    main()
