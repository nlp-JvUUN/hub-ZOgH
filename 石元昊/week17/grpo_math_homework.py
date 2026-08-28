# -*- coding: utf-8 -*-
"""
Week17 作业：基于 GRPO 的强化学习提升模型数学题能力（纯 PyTorch 手写版）
==========================================================================

参考课程项目 `week17 强化学习/grpo_arithmetic`（TRL GRPOTrainer 版，跑在 CUDA 上），
本作业在本机（Apple Silicon MPS，无 trl 依赖）用**纯 PyTorch 手写 GRPO** 复现同一实验：

    probe（基线摸底） → train（手写 GRPO 训练） → probe（复测） → compare（对比表+曲线）

与 TRL 实现的对应关系：
    GRPOTrainer 的组采样        → generate(num_return_sequences=K)
    reward_funcs                → reward_loose_correct(1.0) + reward_format(0.2)
    组内优势归一化              → adv = (r - mean_g) / (std_g + eps)，无价值网络、无奖励模型
    beta=0 不带参考模型         → 同样省略 KL 项
    PPO-clip                    → ratio = exp(logp - old_logp)，min(ratio·A, clip(ratio)·A)

用法（在任意目录执行均可，路径基于脚本位置解析）：
    python grpo_math_homework.py run-all            # 一键全流程（默认规模）
    python grpo_math_homework.py probe --n 4 --k 4  # 单独摸底（快速）
    python grpo_math_homework.py train --steps 2    # 单步冒烟测试
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 路径与全局配置（全部基于脚本位置解析，换目录执行不会出错） ──────────────
ROOT = Path(__file__).resolve().parent                       # homework/
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
# 课程仓库结构：badou/week17 强化学习/homework/grpo_math_homework.py → parents[2] = badou
MODEL_PATH = ROOT.parents[1] / "pretrain_models" / "Qwen2-0.5B-Instruct"

SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")

# 训练集难度配比：按课程方法论（informative group rate 选题），用本机基线摸底实测校准。
# 本仓库的 Qwen2-0.5B-Instruct 权重明显强于课程 CUDA 环境的权重（L3=0.90 / L5=0.80 已饱和），
# 课程的 L3/L5 配比在本机不再处于可学习甜区。实测校准结果（n=10, K=8）：
#   L6 两位×两位     : greedy=0.30, informative=0.10（多数组全错，信号稀）
#   L7 三位×一位     : greedy=0.90(高估), sample=0.79, informative=0.60  ← 甜区
#   L8 两式混合      : greedy=0.00, informative=0.00（能力边界，学不动）
# 因此主训 L7（60%）+ L6（40%）：L7 提供主要梯度信号，L6 提供少量边界内信号。
LEVEL_MIX = [
    ("L7_mul_3x1digit", 0.60),
    ("L6_mul_2x2digit", 0.40),
]
LEVELS = [
    "L1_add_1digit",
    "L2_addsub_2digit",
    "L3_addsub_3digit",
    "L4_mul_1digit",
    "L5_mul_2x1digit",
    "L6_mul_2x2digit",
    "L7_mul_3x1digit",
    "L8_mul_2x2_plus",
]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ---------------------------------------------------------------------------
# 1. 数据：算术题生成 + 输出解析（与课程项目 probe_baseline.py 同构）
# ---------------------------------------------------------------------------
def make_problem(level: str, rng: random.Random):
    """按难度级别生成一道算术题，返回 (表达式文本, 标准答案)。"""
    if level == "L1_add_1digit":        # 个位数加法：sanity check
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"{a} + {b}", a + b
    if level == "L2_addsub_2digit":     # 两位数加减
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)     # 保证减法结果非负
        return f"{a} - {b}", a - b
    if level == "L3_addsub_3digit":     # 三位数加减
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    if level == "L4_mul_1digit":        # 表内乘法
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        return f"{a} × {b}", a * b
    if level == "L5_mul_2x1digit":      # 两位数×一位数
        a, b = rng.randint(10, 99), rng.randint(3, 9)
        return f"{a} × {b}", a * b
    if level == "L6_mul_2x2digit":      # 两位数×两位数
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a} × {b}", a * b
    if level == "L7_mul_3x1digit":      # 三位数×一位数：本机实测甜区（informative=0.60）
        a, b = rng.randint(100, 999), rng.randint(3, 9)
        return f"{a} × {b}", a * b
    if level == "L8_mul_2x2_plus":      # 两位×两位再加三位数：本机实测能力边界（全错）
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(100, 999)
        return f"{a} × {b} + {c}", a * b + c
    raise ValueError(level)


def parse_output(text: str, answer: int):
    """解析模型输出，返回 (是否符合格式, 严格正确, 宽松正确)。

    宽松口径：有 <answer> 标签取标签内数字，否则取输出中最后一个数字。
    基线期模型完全不输出标签（格式率≈0），若只用严格口径，正确信号也是 0，
    组内全零 → advantage 全 0 → GRPO 冷启动失败，所以正确分必须用宽松解析。
    """
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer
    return fmt_ok, strict_ok, loose_ok


def build_prompt_text(tokenizer, expr: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"计算：{expr} = ?"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(model_path, device):
    """加载模型。注意：本地 Qwen2-0.5B 的 config 写着 float16，
    必须显式指定 dtype（课程踩坑 #4：按 config 加载 fp16 → AdamW eps=1e-8 溢出 → 一步训废）。
    本作业实测复现了该坑：MPS + fp16 训练第 1 步即出现 entropy=nan、全部权重损坏；
    fp32 下 AdamW 数值稳定，速度问题用生成批并行缓解。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. probe：基线摸底 / 训练后复测（greedy + pass@k + 格式率 + informative rate）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=8, max_new_tokens=48):
    """分批生成；do_sample=True 时每条 prompt 返回 k 个样本（连续排列）。"""
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0 if do_sample else None,
            top_p=1.0 if do_sample else None,
            num_return_sequences=k if do_sample else 1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:
            all_outputs.extend(decoded[j * k:(j + 1) * k] for j in range(len(batch)))
        else:
            all_outputs.extend(decoded)
    return all_outputs


def cmd_probe(args):
    """基线摸底/复测：输出各难度的格式率、正确率（严格/宽松）、pass@k、informative rate。"""
    model_path = args.model or str(MODEL_PATH)
    print(f"[probe] 加载模型：{model_path}（device={DEVICE}）")
    model, tokenizer = load_model(model_path, DEVICE)
    model.eval()

    rng = random.Random(args.seed)
    report = {"_meta": {"model": model_path, "n": args.n, "k": args.k, "seed": args.seed}}

    for level in LEVELS:
        t0 = time.time()
        problems = [make_problem(level, rng) for _ in range(args.n)]
        texts = [build_prompt_text(tokenizer, expr) for expr, _ in problems]

        # greedy：确定性能力 + 格式遵循
        greedy_outs = generate(model, tokenizer, texts, do_sample=False)
        g_fmt = g_strict = g_loose = 0
        for (expr, ans), out in zip(problems, greedy_outs):
            fmt, strict, loose = parse_output(out, ans)
            g_fmt += fmt
            g_strict += strict
            g_loose += loose

        # 温度采样 K 条：pass@k 与 informative group rate（GRPO 的可学习甜区指标）
        sample_outs = generate(model, tokenizer, texts, do_sample=True, k=args.k)
        s_loose_sum = loose_pass = loose_mixed = 0
        for (_, ans), outs in zip(problems, sample_outs):
            n_loose = sum(parse_output(o, ans)[2] for o in outs)
            s_loose_sum += n_loose
            loose_pass += n_loose > 0
            loose_mixed += 0 < n_loose < args.k
        n = args.n
        report[level] = {
            "greedy_format_rate": round(g_fmt / n, 4),
            "greedy_strict_acc": round(g_strict / n, 4),
            "greedy_loose_acc": round(g_loose / n, 4),
            "sample_loose_acc": round(s_loose_sum / (n * args.k), 4),
            f"loose_pass@{args.k}": round(loose_pass / n, 4),
            "loose_informative_group_rate": round(loose_mixed / n, 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "examples": [
                {"expr": expr, "answer": ans, "greedy_output": out}
                for (expr, ans), out in list(zip(problems, greedy_outs))[:3]
            ],
        }
        r = report[level]
        print(f"  {level:<20} greedy_loose={r['greedy_loose_acc']:.2f} "
              f"fmt={r['greedy_format_rate']:.2f} "
              f"pass@{args.k}={r[f'loose_pass@{args.k}']:.2f} "
              f"informative={r['loose_informative_group_rate']:.2f} "
              f"({r['elapsed_sec']}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[probe] 结果已保存：{out_path}")


# ---------------------------------------------------------------------------
# 3. train：手写 GRPO（无价值网络、无奖励模型、beta=0 无参考模型）
# ---------------------------------------------------------------------------
def reward_of(text: str, answer: int):
    """复合奖励（与课程项目一致）：宽松正确分 1.0 + 格式分 0.2。"""
    fmt, _, loose = parse_output(text, answer)
    return (1.0 if loose else 0.0) + (0.2 if fmt else 0.0), loose, fmt


def per_token_logprob(model, input_ids, attention_mask, completion_mask):
    """计算 completion 部分的逐 token log 概率，返回 [B, T]（padding 位为 0）。

    completion_mask: [B, T]，与 input_ids 右对齐的 completion 区段标记。
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # 位置 t 的 token 由位置 t-1 的 logits 预测
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    mask = completion_mask[:, 1:].to(token_lp.dtype)
    return token_lp * mask, mask


def cmd_train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[train] 加载基座：{MODEL_PATH}（device={DEVICE}）")
    model, tokenizer = load_model(MODEL_PATH, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    train_rng = random.Random(args.data_seed)
    log_rows = []
    t_start = time.time()

    for step in range(1, args.steps + 1):
        # ── 3.1 采样阶段：P 个 prompt × K 条组内采样（GRPO 的"训练燃料"） ──
        problems, prompt_texts = [], []
        for _ in range(args.prompts_per_step):
            r, acc = train_rng.random(), 0.0
            level = LEVEL_MIX[-1][0]
            for lv, p in LEVEL_MIX:
                acc += p
                if r <= acc:
                    level = lv
                    break
            expr, ans = make_problem(level, train_rng)
            problems.append((expr, ans))
            prompt_texts.append(build_prompt_text(tokenizer, expr))

        model.eval()
        with torch.no_grad():
            enc = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(DEVICE)
            prompt_len = enc["input_ids"].shape[1]
            # 把每条 prompt 显式复制 K 份拼成大批：MPS 上批并行吞吐远高于
            # num_return_sequences 的内部扩展，是 fp32 下主要的提速手段；
            # 注意结果按 [prompt0×K, prompt1×K, ...] 连续排列，与 pairs 对齐
            flat_ids = enc["input_ids"].repeat_interleave(args.k, dim=0)
            flat_mask = enc["attention_mask"].repeat_interleave(args.k, dim=0)
            gen_out = model.generate(
                input_ids=flat_ids, attention_mask=flat_mask,
                max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        comp_ids = gen_out[:, prompt_len:]                       # [P*K, T_c]
        comp_texts = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)

        # ── 3.2 奖励与组内优势归一化（无价值网络：用组内均值/标准差代替） ──
        n_correct = n_fmt = 0
        pairs = [(t, problems[i // args.k][1]) for i, t in enumerate(comp_texts)]
        r_vals, loose_flags, fmt_flags = [], [], []
        for text, ans in pairs:
            r, loose, fmt = reward_of(text, int(ans))
            r_vals.append(r)
            loose_flags.append(loose)
            n_fmt += fmt
            n_correct += loose
        r_t = torch.tensor(r_vals, dtype=torch.float32).view(args.prompts_per_step, args.k)
        g_mean, g_std = r_t.mean(dim=1, keepdim=True), r_t.std(dim=1, keepdim=True)
        zero_std = (g_std.squeeze(-1) == 0)                      # 全对/全错组：无学习信号
        adv = torch.where(g_std > 0, (r_t - g_mean) / (g_std + 1e-6), torch.zeros_like(r_t))
        adv = adv.view(-1).to(DEVICE)                            # [P*K]

        # ── 3.3 训练阶段：old logprob（stop-grad）→ PPO-clip 目标 → 一步更新 ──
        full_ids = gen_out                                       # prompt+completion，右对齐
        full_mask = (full_ids != tokenizer.pad_token_id).long()
        comp_mask_full = torch.zeros_like(full_mask)
        comp_mask_full[:, prompt_len:] = (comp_ids != tokenizer.pad_token_id).long()

        model.eval()
        with torch.no_grad():
            old_lp, tok_mask = per_token_logprob(model, full_ids, full_mask, comp_mask_full)
            # 策略熵（训练动态晴雨表之一）：completion token 的平均熵
            logits = model(input_ids=full_ids, attention_mask=full_mask).logits[:, :-1, :]
            p = logits.softmax(-1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
            entropy = (ent * tok_mask).sum() / tok_mask.sum().clamp_min(1)

        model.train()
        lp, tok_mask = per_token_logprob(model, full_ids, full_mask, comp_mask_full)
        ratio = torch.exp(lp - old_lp)                           # 同批采样：初始恒为 1
        adv_b = adv.unsqueeze(-1).expand_as(lp)
        surr1 = ratio * adv_b
        surr2 = ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * adv_b
        loss = -torch.min(surr1, surr2)
        loss = (loss * tok_mask).sum() / tok_mask.sum().clamp_min(1)   # 按有效 token 平均

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── 3.4 记录 ──
        row = {
            "step": step,
            "reward_mean": round(float(r_t.mean()), 4),
            "loose_acc": round(n_correct / len(comp_texts), 4),
            "format_rate": round(n_fmt / len(comp_texts), 4),
            "frac_zero_std": round(float(zero_std.float().mean()), 4),
            "entropy": round(float(entropy), 4),
            "loss": round(float(loss.detach()), 6),
            "grad_norm": round(float(grad_norm), 4),
        }
        log_rows.append(row)
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>3}/{args.steps}  reward={row['reward_mean']:.3f} "
                  f"acc={row['loose_acc']:.2f} fmt={row['format_rate']:.2f} "
                  f"zero_std={row['frac_zero_std']:.2f} ent={row['entropy']:.3f} "
                  f"grad={row['grad_norm']:.2f}")

    # ── 3.5 保存：训练日志 + 训练后模型 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": log_rows,
                   "total_sec": round(time.time() - t_start, 1)}, f,
                  ensure_ascii=False, indent=2)
    ckpt_dir = OUT_DIR / "grpo_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[train] 日志：{log_path}\n[train] 模型已保存：{ckpt_dir} "
          f"（总耗时 {time.time() - t_start:.0f}s）")


# ---------------------------------------------------------------------------
# 4. compare：前后对比表 + 训练曲线
# ---------------------------------------------------------------------------
def cmd_compare(args):
    with open(args.before, encoding="utf-8") as f:
        before = json.load(f)
    with open(args.after, encoding="utf-8") as f:
        after = json.load(f)
    with open(args.train_log, encoding="utf-8") as f:
        train_log = json.load(f)

    print("\n===== GRPO 训练前后对比（同评估协议，同 seed） =====")
    k = before["_meta"]["k"]
    header = (f"{'难度':<18} {'在训练集':^6} | {'格式率 前→后':^14} "
              f"{'greedy宽松正确 前→后':^20} {'pass@%d 前→后' % k:^16}")
    print(header)
    print("-" * 78)
    train_levels = {lv for lv, _ in LEVEL_MIX}
    for level in LEVELS:
        b, a = before[level], after[level]
        pk = f"loose_pass@{k}"
        mark = "√" if level in train_levels else "—"
        print(f"{level:<18} {mark:^6} | "
              f"{b['greedy_format_rate']:.2f} → {a['greedy_format_rate']:.2f}    "
              f"{b['greedy_loose_acc']:.2f} → {a['greedy_loose_acc']:.2f}      "
              f"{b[pk]:.2f} → {a[pk]:.2f}")

    # ── 训练曲线（英文标签，避免中文缺字体显示为方框） ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = train_log["rows"]
    steps = [r["step"] for r in rows]
    panels = [
        ("reward_mean", "mean reward"), ("loose_acc", "loose accuracy"),
        ("format_rate", "format rate"), ("frac_zero_std", "frac zero-std groups"),
        ("entropy", "policy entropy"), ("grad_norm", "grad norm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(steps, [r[key] for r in rows], lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle("Handwritten GRPO training on Qwen2-0.5B arithmetic")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "train_curves.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\n[compare] 曲线已保存：{fig_path}")


# ---------------------------------------------------------------------------
# 5. 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO 数学题强化学习作业（手写版）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="基线摸底 / 训练后复测")
    p.add_argument("--model", type=str, default=None, help="模型路径；默认基座，复测时传 checkpoint")
    p.add_argument("--out", type=str, default=str(OUT_DIR / "baseline_probe.json"))
    p.add_argument("--n", type=int, default=10, help="每个难度的题目数")
    p.add_argument("--k", type=int, default=8, help="pass@k 采样数")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="手写 GRPO 训练")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--prompts-per-step", type=int, default=4, dest="prompts_per_step")
    p.add_argument("--k", type=int, default=8, help="组内采样数（GRPO group size，与课程一致）")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO-clip 范围")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=48, dest="max_new_tokens")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-seed", type=int, default=123, dest="data_seed")
    p.add_argument("--log-every", type=int, default=5, dest="log_every")

    p = sub.add_parser("compare", help="前后对比表 + 训练曲线")
    p.add_argument("--before", type=str, default=str(OUT_DIR / "baseline_probe.json"))
    p.add_argument("--after", type=str, default=str(OUT_DIR / "post_probe.json"))
    p.add_argument("--train-log", type=str, default=str(OUT_DIR / "train_log.json"),
                   dest="train_log")

    sub.add_parser("run-all", help="一键全流程：probe → train → probe → compare")

    args = parser.parse_args()
    if args.cmd == "run-all":
        args.model = None
        args.out = str(OUT_DIR / "baseline_probe.json")
        args.n, args.k, args.seed = 10, 8, 42
        cmd_probe(args)
        args = argparse.Namespace(
            steps=40, prompts_per_step=2, k=8, lr=1e-6, epsilon=0.2,
            temperature=1.0, max_new_tokens=48, seed=7, data_seed=123, log_every=5)
        cmd_train(args)
        args = argparse.Namespace(
            model=str(OUT_DIR / "grpo_ckpt"), out=str(OUT_DIR / "post_probe.json"),
            n=10, k=8, seed=42)
        cmd_probe(args)
        args = argparse.Namespace(
            before=str(OUT_DIR / "baseline_probe.json"),
            after=str(OUT_DIR / "post_probe.json"),
            train_log=str(OUT_DIR / "train_log.json"))
        cmd_compare(args)
    elif args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
