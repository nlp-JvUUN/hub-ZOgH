# -*- coding: utf-8 -*-
"""
evaluate.py —— 基线/训练后统一评估（同一脚本，--model 切换）
==============================================================
指标（与课件 grpo_arithmetic 对齐）：
  greedy 正确率     —— 确定性解码，测"聚拢后的能力"
  greedy 格式率     —— 输出是否带 <answer>N</answer>
  pass@8            —— 温度 1.0 采 8 条，任一正确（测"分布里有没有答案"）
  informative rate  —— 每题 K 采中 0<正确<K 的比例（GRPO 可学习性的晴雨表）
"""
import argparse
import json
import os

import torch

from tinygpt import TinyGPT, decode, prompt_tensor

PROMPT_MAX = 24
MAX_NEW = 24


def parse_answer(text):
    """宽松解析：有 <answer> 标签取标签内数字，否则取第一个数字。
    注意：小模型在温度采样下偶尔会在正确答案后继续输出数字，
    取"第一个数字"比"最后一个数字"对本任务更稳健。"""
    import re
    m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    return int(nums[0]) if nums else None


def has_format(text):
    import re
    return bool(re.search(r"<answer>\s*-?\d+\s*</answer>", text))


def eval_level(model, items, device, K=8, temp=1.0):
    """items: [(question, answer)]；返回该级各项指标。
    采样方式：一次生成 B 题 × K 条 = B*K 条，按题分组统计。"""
    qs = [q + "\nA: " for q, _ in items]
    ans = [int(a) for _, a in items]
    B = 8  # 每批题数
    n = len(qs)

    # ---- greedy ----
    g_correct = g_format = 0
    for i in range(0, n, B):
        pt, pm = prompt_tensor(qs[i:i + B], device=device)
        comps, _, _ = model.generate(pt, pm, max_new=MAX_NEW, greedy=True)
        for c, a in zip(comps, ans[i:i + B]):
            t = decode(c.tolist())
            g_correct += parse_answer(t) == a
            g_format += has_format(t)

    # ---- pass@K 与 informative rate（同一批采样数据）----
    ok_any = [0] * n          # 每题 K 条里是否至少一条正确
    n_correct = [0] * n       # 每题 K 条里正确条数
    for i in range(0, n, B):
        qb = qs[i:i + B]
        ab = ans[i:i + B]
        pt, pm = prompt_tensor(qb, device=device)
        pts = pt.repeat_interleave(K, dim=0)          # [B*K, Lp]
        pms = pm.repeat_interleave(K, dim=0)
        comps, _, _ = model.generate(pts, pms, max_new=MAX_NEW, temperature=temp)
        for j, c in enumerate(comps):
            t = decode(c.tolist())
            ok = parse_answer(t) == ab[j // K]
            if ok:
                ok_any[i + j // K] = 1
                n_correct[i + j // K] += 1

    n_inform = sum(1 for c in n_correct if 0 < c < K)

    return {
        "n": n,
        "greedy_acc": g_correct / n,
        "greedy_format": g_format / n,
        f"pass@{K}_acc": sum(ok_any) / n,
        "informative_rate": n_inform / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../outputs/sft_ckpt/sft.pt")
    ap.add_argument("--out", default="../outputs/baseline_probe.json")
    ap.add_argument("--per_level", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch as _torch
    _torch.manual_seed(args.seed)
    from data import LEVELS, make_eval_set
    device = args.device
    model = TinyGPT.load(args.model, device)
    model.eval()

    evalset = make_eval_set(per_level=args.per_level, seed=args.seed)
    result = {"model": args.model, "seed": args.seed}
    for lv in LEVELS:
        r = eval_level(model, evalset[lv], device)
        result[lv] = r
        print(f"[{lv}] greedy_acc {r['greedy_acc']:.2f} | greedy_format "
              f"{r['greedy_format']:.2f} | pass@8 {r['pass@8_acc']:.2f} | "
              f"informative {r['informative_rate']:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
