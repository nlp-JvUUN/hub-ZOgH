"""
grpo_loop.py — 自实现的最小 GRPO 训练循环（不依赖 TRL）

为什么不用 TRL：
  当前环境是 transformers 5.13.0（比 trl 0.21 所适配的 5.5.3 更新），两者存在版本兼容地雷。
  本作业要求"独立可提交、可解释"，因此用纯 torch 从零实现一个最小但正确的 GRPO。

GRPO 数学（beta=0，无参考模型 / 无 KL 项）：
  每个优化步：
    1. 对一批 prompt，各采样 K 条 completion（旧策略 π_old），记录采样时的 old_log_prob。
    2. 组内归一化：A_i = (r_i - mean(r)) / (std(r) + eps)   ← 替代 Critic 价值网络。
    3. 对这批采样做多个 epoch 的梯度更新：
         ratio_i   = exp(log π_θ(c_i) - log π_old(c_i))
         surrogate = min(ratio_i·A_i, clip(ratio_i, 1-ε, 1+ε)·A_i)
         loss      = -mean(surrogate)   （对组内所有 completion 平均）

  说明：
    - old_log_prob 在“采样时”冻结；每个 epoch 用当前 θ 重算 new_log_prob → ratio ≠ 1，
      从而 clip 真正生效（这正是 GRPO/PPO 与朴素策略梯度的区别）。
    - 同一 prompt 组内共享同一基准（mean/std），完成组内对比。
    - 序列级实现：对每条 completion 用“生成出的 token”求平均 log-prob 作为序列似然代理
      （严格实现应对生成 token 求和；序列短且为教学演示，用均值已足够展示机制与正确梯度方向）。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW


def _gen_log_prob(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  prompt_len: int, require_grad: bool) -> torch.Tensor:
    """计算一条序列（prompt + 生成部分）在新/旧策略下的平均 token log-prob。

    只用“生成出的 token”计算：把 logits 右移一位，取预测下一个 token 的 log-prob，
    且只取位于 prompt 之后的有效非 pad 位置。返回该序列的平均 log-prob（标量张量）。
    require_grad=True 时保留计算图（用于训练），否则用 no_grad（用于采样时冻结）。
    """
    ctx = torch.no_grad if not require_grad else torch.enable_grad
    with ctx():
        logits = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # [T, V]
    gen_idx = torch.arange(prompt_len, input_ids.size(0), device=input_ids.device)
    target = input_ids[gen_idx]
    gathered = log_probs[gen_idx - 1, target]  # [len(gen_idx)]
    valid = attention_mask[gen_idx].bool()
    if valid.sum() == 0:
        return torch.zeros((), device=input_ids.device)
    return gathered[valid].mean()


def grpo_step(model: nn.Module, optimizer: AdamW, prompts: List[str], answers: List[int],
              tokenizer, reward_correct: Callable, reward_format: Callable,
              num_generations: int = 8, epsilon: float = 0.2, temperature: float = 1.0,
              max_new_tokens: int = 64, epochs: int = 4, log_completions: bool = False) -> Dict[str, float]:
    """执行一个 GRPO 优化步（对这批 prompt 采样一次，再做 epochs 次梯度更新）。

    返回聚合指标：{pg_loss, reward_mean, surrogate_mean, n_groups}。
    """
    device = next(model.parameters()).device
    model.train()

    # ── 1. 采样 + 记录旧策略 log-prob ──────────────────────────────────────
    batch_data = []  # 每项：{input_ids, prompt_len, advantage, old_lp, reward}
    all_rewards: List[float] = []
    for p_text, p_answer in zip(prompts, answers):
        enc = tokenizer(p_text, return_tensors="pt", padding=False).to(device)
        prompt_len = enc["input_ids"].size(1)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                num_return_sequences=num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = [tokenizer.decode(gen[j][prompt_len:], skip_special_tokens=True) for j in range(num_generations)]
        r_correct = reward_correct(texts, [p_answer] * num_generations)
        r_format = reward_format(texts)
        rewards = [c + f for c, f in zip(r_correct, r_format)]
        all_rewards.extend(rewards)

        mean_r = sum(rewards) / len(rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        for j in range(num_generations):
            seq = gen[j]
            attn = torch.ones_like(seq)
            old_lp = _gen_log_prob(model, seq, attn, prompt_len, require_grad=False)
            batch_data.append({
                "input_ids": seq,
                "attention_mask": attn,
                "prompt_len": prompt_len,
                "advantage": advantages[j],
                "old_lp": old_lp,
                "reward": rewards[j],
            })
            if log_completions:
                print(f"    comp{j}: {texts[j]!r}  r={rewards[j]:.2f} adv={advantages[j]:.3f} old_lp={old_lp.item():.3f}")

    n = len(batch_data)
    # ── 2. 对这批采样做 epochs 次梯度更新 ──────────────────────────────────
    total_pg = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        pg_loss = torch.zeros((), device=device)
        for item in batch_data:
            new_lp = _gen_log_prob(model, item["input_ids"], item["attention_mask"],
                                   item["prompt_len"], require_grad=True)
            ratio = torch.exp(new_lp - item["old_lp"].detach())
            adv = torch.tensor(item["advantage"], device=device)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            surrogate = torch.min(unclipped, clipped)
            pg_loss = pg_loss - surrogate  # 最大化 surrogate → 最小化其负值
        pg_loss = pg_loss / n
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_pg += pg_loss.item()

    model.eval()
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    return {
        "pg_loss": total_pg / max(epochs, 1),
        "surrogate": -total_pg / max(epochs, 1),
        "reward_mean": mean_reward,
        "n_groups": len(prompts),
        "n_completions": n,
    }
