# -*- coding: utf-8 -*-
"""
train_grpo.py —— 阶段二：从零实现 GRPO（不依赖 TRL）
======================================================
对照课件第四部分「DeepSeek GRPO」：
  1. 组相对策略优化：同一 prompt 采 K 条回答，用组内平均奖励代替 Critic
  2. 优势 A_i = (r_i - mean(r)) / std(r)，组内归一化（std=0 的组不产生梯度）
  3. PPO-Clip 目标：L = E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]
  4. KL 正则：β·KL(π_θ || π_ref)，参考模型 = SFT 模型（冻结），防漂移
  5. 奖励 = 答案正确(1.0) + 输出格式(0.2)，全部可程序化验证

每步 = n_prompts 道题 × K 条采样 = 一组"组内相对比较"。
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from tinygpt import PAD, STOI, GPTConfig, TinyGPT, decode, encode, prompt_tensor

PROMPT_MAX = 24
MAX_NEW = 24
BETA = 0.05          # KL 惩罚系数
EPS = 0.2            # PPO clip 范围
TEMP = 1.0           # 采样温度（组内多样性的来源）
WEIGHT_FORMAT = 0.2  # 格式分权重（故意小于正确分：主次信号竞争，教学点）


def parse_answer(text):
    """宽松解析：有 <answer> 标签取标签内数字，否则取最后一个数字。"""
    import re
    m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    return int(nums[0]) if nums else None  # 取第一个数字（见 evaluate.py 注释）


def has_format(text):
    import re
    return bool(re.search(r"<answer>\s*-?\d+\s*</answer>", text))


def reward_batch(completions, answers):
    """completions: list[str]；answers: list[str]。返回 (correct, format)。"""
    correct, fmt = [], []
    for comp, ans in zip(completions, answers):
        correct.append(1.0 if parse_answer(comp) == int(ans) else 0.0)
        fmt.append(1.0 if has_format(comp) else 0.0)
    return torch.tensor(correct), torch.tensor(fmt)


def group_advantages(rewards, K):
    """组内归一化优势：A = (r - mean) / (std + 1e-4)；std=0 的组全置 0。"""
    r = rewards.view(-1, K)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    adv = (r - mean) / (std + 1e-4)
    zero = (std < 1e-4).expand_as(adv)
    adv[zero] = 0.0
    return adv.view(-1), (std < 1e-4).view(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n_prompts", type=int, default=8)    # 每步几道题
    ap.add_argument("--K", type=int, default=8)            # 每组采样数
    ap.add_argument("--ppo_epochs", type=int, default=1)   # 每批数据更新几轮
    ap.add_argument("--mini_bs", type=int, default=32)     # 小批量大小
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sft_ckpt", default="../outputs/sft_ckpt/sft.pt")
    ap.add_argument("--out", default="../outputs/grpo_ckpt/grpo.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import random
    torch.manual_seed(args.seed)
    from data import gen_problem, prompt_of
    rng = random.Random(args.seed)
    device = args.device

    policy = TinyGPT.load(args.sft_ckpt, device)
    policy.train()
    ref = TinyGPT.load(args.sft_ckpt, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"策略参数量：{policy.n_params()/1e6:.2f}M（ref 冻结）")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    log = []
    t0 = time.time()

    for step in range(args.steps):
        # ---- 1. 采样 n_prompts 道 L2 题，每组重复 K 次 ----
        prompts, answers = [], []
        for _ in range(args.n_prompts):
            q, a = gen_problem("L2", rng)
            prompts.append(q + "\nA: ")
            answers.append(a)
        pts, pmask = prompt_tensor(prompts, device=device)
        pts_tiled = pts.repeat_interleave(args.K, dim=0)  # [n*K, Lp]
        pmask_tiled = pmask.repeat_interleave(args.K, dim=0)
        answers_tiled = [a for a in answers for _ in range(args.K)]

        # ---- 2. 当前策略采样 K 条补全（T=1.0，遇 EOS 停止），记录 old logprobs ----
        comps, old_logp, cmask = policy.generate(
            pts_tiled, pmask_tiled, max_new=MAX_NEW, temperature=TEMP)
        full = torch.cat([pts_tiled, comps], dim=1)
        full_mask = torch.cat([pmask_tiled, cmask], dim=1)
        Lp = pts_tiled.shape[1]
        valid = cmask.float()                             # [n*K, Lc]

        # ---- 3. 规则奖励：正确(1.0) + 格式(0.2) ----
        texts = [decode(c.tolist()) for c in comps]
        correct, fmt = reward_batch(texts, answers_tiled)
        rewards = correct + WEIGHT_FORMAT * fmt

        # ---- 4. 组内归一化优势 ----
        adv, zero_std = group_advantages(rewards, args.K)

        # ---- 5. 策略更新：同一批数据上做 PPO 多轮小批量更新 ----
        # （每轮更新前重算 new logprobs，ratio 对"旧策略"比较，clip 防步子过大）
        with torch.no_grad():
            ref_logp, _ = ref.completion_logprobs(full, Lp, full_mask)
        idx = list(range(len(texts)))
        pg_losses, kl_means, ent_means, clip_fracs = [], [], [], []
        for epoch_i in range(args.ppo_epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), args.mini_bs):
                mb = idx[s:s + args.mini_bs]
                logits = policy(full[mb], full_mask[mb])
                new_logp_all = F.log_softmax(logits, dim=-1)
                tok = full[mb, Lp:]
                new_tok = new_logp_all[:, Lp - 1:-1].gather(2, tok.unsqueeze(2)).squeeze(2)
                v = valid[mb]
                ratio = torch.exp(new_tok - old_logp[mb])
                pg1 = ratio * adv[mb].unsqueeze(1)
                pg2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv[mb].unsqueeze(1)
                pg_loss = -torch.min(pg1, pg2)
                kl = torch.exp(ref_logp[mb] - new_tok) - (ref_logp[mb] - new_tok) - 1
                loss = ((pg_loss + args.beta * kl) * v).sum() / v.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    pg_losses.append(((pg_loss * v).sum() / v.sum()).item())
                    kl_means.append(((kl * v).sum() / v.sum()).item())
                    ent_means.append(((-(new_logp_all[:, Lp - 1:-1].exp()
                                         * new_logp_all[:, Lp - 1:-1]).sum(-1) * v).sum()
                                      / v.sum()).item())
                    clip_fracs.append((((pg2 < pg1) * v).sum() / v.sum()).item())

        # ---- 6. 指标 ----
        log.append({
            "step": step,
            "reward": rewards.mean().item(),
            "reward_correct": correct.mean().item(),
            "reward_format": fmt.mean().item(),
            "adv_std": adv.std().item(),
            "frac_zero_std": zero_std.float().mean().item(),
            "kl": sum(kl_means) / len(kl_means),
            "entropy": sum(ent_means) / len(ent_means),
            "clip_frac": sum(clip_fracs) / len(clip_fracs),
            "grad_norm": float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)),
        })
        if step % 5 == 0 or step == args.steps - 1:
            e = log[-1]
            print(f"step {step:3d} | 正确 {e['reward_correct']:.2f} "
                  f"格式 {e['reward_format']:.2f} | 零方差组 {e['frac_zero_std']:.2f} "
                  f"| KL {e['kl']:.3f} 熵 {e['entropy']:.3f} | {time.time()-t0:.0f}s")

    # ---- 7. 存盘 ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    policy.save(args.out)
    tag = f"_{args.tag}" if args.tag else ""
    log_path = f"../outputs/train_log{tag}.json"
    with open(log_path, "w") as f:
        json.dump({"config": vars(args), "log": log}, f, indent=2)
    print(f"GRPO 完成：{time.time()-t0:.0f}s → {args.out} + {log_path}")

    # 打印 3 个训练前后样例
    policy.eval()
    for _ in range(3):
        q, a = gen_problem("L2", rng)
        pt, pm = prompt_tensor([q + "\nA: "], device=device)
        comp, _, _ = policy.generate(pt, pm, max_new=MAX_NEW, greedy=True)
        print(f"  {q} -> {decode(comp[0].tolist())}  (真值 {a})")


if __name__ == "__main__":
    main()
