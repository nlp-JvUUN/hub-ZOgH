# -*- coding: utf-8 -*-
"""
train_sft.py —— 阶段一：SFT，得到 GRPO 的起点策略
==================================================
固定数据、固定 seed、固定步数 —— 全程可复现。
校准方法：用 --probe 在训练结束时打印各难度 greedy 正确率，
把 --steps 停在 L2 的"能力相变中段"（基线 ≈ 50-80%，GRPO 才有提升空间）。
"""
import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F

from tinygpt import (BOS, EOS, PAD, STOI, GPTConfig, TinyGPT, decode, encode,
                     prompt_tensor)

BLOCK = 64


def build_batch(data, idx, batch_size, device):
    """把 (level, question, answer) 拼成完整序列：BOS + prompt + answer + EOS。"""
    seqs = []
    for lv, q, a in data[idx:idx + batch_size]:
        seqs.append([STOI[BOS]] + encode(q + "\nA: ") + encode(a) + [STOI[EOS]])
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), STOI[PAD], dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    mask = (ids != STOI[PAD]).long()
    return ids.to(device), mask.to(device)


@torch.no_grad()
def greedy_acc(model, items, device, max_new=16):
    """items: [(question, answer)]；greedy 正确率。"""
    model.eval()
    qs = [q + "\nA: " for q, _ in items]
    ans = [int(a) for _, a in items]
    ok = 0
    for i in range(0, len(qs), 32):
        pt, pm = prompt_tensor(qs[i:i + 32], device=device)
        comps, _, _ = model.generate(pt, pm, max_new=max_new, greedy=True)
        for c, a in zip(comps, ans[i:i + 32]):
            nums = re.findall(r"-?\d+", decode(c.tolist()))
            ok += (int(nums[0]) if nums else None) == a
    model.train()
    return ok / len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=0,
                    help="总优化步数；0 = 自动（4 个 epoch）")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="../outputs/sft_ckpt/sft.pt")
    ap.add_argument("--probe", action="store_true",
                    help="训练结束时打印各难度 greedy 正确率（校准用）")
    args = ap.parse_args()
    device = args.device

    from data import LEVELS, make_eval_set, make_sft_dataset
    data = make_sft_dataset(args.n_data, seed=0)
    rng = __import__("random").Random(0)
    rng.shuffle(data)
    steps_per_epoch = args.n_data // args.batch_size
    total_steps = args.steps or steps_per_epoch * 4
    print(f"语料 {len(data)} 条 / batch {args.batch_size} / 总步数 {total_steps}")

    torch.manual_seed(0)
    cfg = GPTConfig(n_embd=192, n_layer=6)   # ~2.7M 参数，CPU 可训
    model = TinyGPT(cfg).to(device)
    print(f"模型参数量：{model.n_params() / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log = []
    t0 = time.time()
    for step in range(total_steps):
        idx = (step % steps_per_epoch) * args.batch_size
        ids, mask = build_batch(data, idx, args.batch_size, device)
        logits = model(ids, mask)
        logp = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(logp[:, :-1].reshape(-1, logp.size(-1)),
                          ids[:, 1:].reshape(-1), reduction="none")
        loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == total_steps - 1:
            log.append({"step": step, "loss": loss.item()})
            print(f"step {step:4d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    with open("../outputs/sft_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"SFT 完成：{time.time()-t0:.0f}s → {args.out}")

    if args.probe:
        evalset = make_eval_set(per_level=200, seed=42)
        for lv in LEVELS:
            acc = greedy_acc(model, evalset[lv], device)
            print(f"[{lv}] greedy_acc {acc:.2f}", flush=True)
        for lv, q, a in [evalset["L2"][0], evalset["L2"][1], evalset["L4"][0]]:
            pt, pm = prompt_tensor([q + "\nA: "], device=device)
            comps, _, _ = model.generate(pt, pm, max_new=16, greedy=True)
            print(f"  样例 {q} -> {decode(comps[0].tolist())!r} (真值 {a})")


if __name__ == "__main__":
    main()
