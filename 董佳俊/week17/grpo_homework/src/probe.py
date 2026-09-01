"""
probe.py — 基线摸底 / 训练后评估（同一脚本，--model 切换）

教学重点：
  1. GRPO 的"可学习甜区"：一个 prompt 采样 K 条，组内有对有错才有非零 advantage；
     全对或全错的组不产生梯度 —— informative group rate 是选题难度的核心指标。
  2. greedy 正确率 vs pass@k 差异 → 采样多样性是 GRPO 的训练燃料。
  3. 训练前后用相同 seed 生成相同题目，才能做配对比较。

输出（JSON）：
  baseline_probe.json / post_probe.json：各难度的 greedy/采样/pass@k/格式率/informative rate。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import LEVELS, SYSTEM_PROMPT, make_problem, parse_output

# 默认模型路径：指向当前机器上存在的那份 Qwen2-0.5B-Instruct
DEFAULT_MODEL = r"D:\八斗学习内容\pretrain_models\Qwen2-0.5B-Instruct"


def _build_prompt(tokenizer, expr: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"计算：{expr} = ?"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _generate(model, tokenizer, texts: List[str], do_sample: bool, k: int = 8,
              batch_size: int = 8, max_new_tokens: int = 64) -> List[List[str]]:
    """分批生成。do_sample=True 时每条 prompt 返回 k 条样本；外层按 prompt 对齐返回。"""
    all_outputs: List[List[str]] = []
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
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:
            for j in range(len(batch)):
                all_outputs.append(decoded[j * k:(j + 1) * k])
        else:
            all_outputs.append(decoded)
    return all_outputs


def load_model(path: str):
    """加载模型；若是 LoRA checkpoint 会自动先加载基座再挂 adapter（保持兼容）。

    device_map 不设为 "auto"：那需要 accelerate（当前环境未装）；本机单设备直接 .to(device)。
    """
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel  # 仅当评估 LoRA 产物时需要
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    return tok, model


def run_probe(model, tokenizer, n: int, k: int, seed: int, device: str) -> Dict:
    rng = random.Random(seed)
    report: Dict = {}
    for level in LEVELS:
        t0 = time.time()
        problems = [make_problem(level, rng) for _ in range(n)]
        texts = [_build_prompt(tokenizer, expr) for expr, _ in problems]

        # 1) greedy：确定性能力 + 格式遵循
        greedy_outs = _generate(model, tokenizer, texts, do_sample=False)[0]
        greedy_fmt = greedy_strict = greedy_loose = 0
        for (_, ans), out in zip(problems, greedy_outs):
            fmt, strict, loose = parse_output(out, ans)
            greedy_fmt += fmt
            greedy_strict += strict
            greedy_loose += loose

        # 2) 采样 k 条：pass@k + informative group rate（严格 / 宽松两套口径）
        sample_outs = _generate(model, tokenizer, texts, do_sample=True, k=k)
        sample_strict_sum = sample_loose_sum = 0
        pass_at_k = loose_pass_at_k = 0
        mixed = loose_mixed = 0
        for (_, ans), outs in zip(problems, sample_outs):
            results = [parse_output(o, ans) for o in outs]
            n_strict = sum(r[1] for r in results)
            n_loose = sum(r[2] for r in results)
            sample_strict_sum += n_strict
            sample_loose_sum += n_loose
            pass_at_k += n_strict > 0
            loose_pass_at_k += n_loose > 0
            mixed += 0 < n_strict < k
            loose_mixed += 0 < n_loose < k

        report[level] = {
            "n": n, "k": k,
            "greedy_format_rate": round(greedy_fmt / n, 4),
            "greedy_strict_acc": round(greedy_strict / n, 4),
            "greedy_loose_acc": round(greedy_loose / n, 4),
            "sample_strict_acc": round(sample_strict_sum / (n * k), 4),
            "sample_loose_acc": round(sample_loose_sum / (n * k), 4),
            f"pass@{k}": round(pass_at_k / n, 4),
            f"loose_pass@{k}": round(loose_pass_at_k / n, 4),
            "informative_group_rate": round(mixed / n, 4),
            "loose_informative_group_rate": round(loose_mixed / n, 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "examples": [
                {"expr": expr, "answer": ans, "greedy_output": out}
                for (expr, ans), out in zip(problems, greedy_outs[:3])
            ],
        }
        r = report[level]
        print(f"{level:<20} greedy_loose={r['greedy_loose_acc']:.2f} fmt={r['greedy_format_rate']:.2f} "
              f"loose_pass@{k}={r[f'loose_pass@{k}']:.2f} loose_informative={r['loose_informative_group_rate']:.2f} "
              f"({r['elapsed_sec']}s)")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型/checkpoint 路径")
    ap.add_argument("--quick", action="store_true", help="每难度只跑 10 题")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="baseline_probe.json", help="结果 JSON 路径（相对项目根或绝对路径）")
    args = ap.parse_args()
    n = 10 if args.quick else args.n

    tokenizer, model = load_model(args.model)
    print(f"device: {next(model.parameters()).device}")

    report = run_probe(model, tokenizer, n, args.k, args.seed, str(next(model.parameters()).device))

    out = Path(args.out)
    if not out.is_absolute() and not str(out).startswith("outputs"):
        out = Path(__file__).parent.parent / "outputs" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
