"""
验证速度提升：三路吞吐对比（最小闭环）

  [A] transformers 串行         —— baseline，一次一条
  [B] transformers batch=8      —— 只加"批处理"（手动 padding）
  [C] vLLM 批处理               —— 批处理 + PagedAttention + continuous batching

同一个模型、同一批 prompts、相同 max_new_tokens，测总耗时 / QPS / tokens/s，
产出 outputs/throughput_results.json + outputs/throughput_comparison.png。

使用方式（⚠️ 先停掉 vLLM server 释放显存，否则两份模型 + KV cache 会 OOM）：
  fuser -k 8000/tcp
  python bench_throughput.py
  快速冒烟：python bench_throughput.py --n 10 --max-tokens 32
"""

import argparse
import gc
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

DEFAULT_MODEL = "/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"


# ── 测试 prompts：长短混合（continuous batching 的收益在长短不均时最明显）──
def build_prompts(n: int) -> list[str]:
    short = ["什么是股票？", "什么是ETF？", "什么是期权？", "什么是PE？", "什么是ROE？"]
    medium = ["解释一下价值投资和趋势投资的区别。", "为什么会出现股市崩盘？",
              "沪深300和中证500有什么区别？", "什么是量化交易？", "可转债有哪些特点？"]
    long = ["请详细介绍一下巴菲特的投资理念及其核心原则，并举例说明。",
            "解释下现金流折现（DCF）估值法的基本步骤、使用的参数以及它的局限性。"]
    pool = short * 3 + medium * 2 + long * 2
    return (pool * ((n + len(pool) - 1) // len(pool)))[:n]


# ══════════════════════════════════════════════════════════════════
# 模式 A+B：transformers
# ══════════════════════════════════════════════════════════════════
def bench_transformers(model_path: str, prompts: list[str],
                       max_new_tokens: int, batch_size: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    def make_chat(q: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

    chat_prompts = [make_chat(q) for q in prompts]

    # [A] 串行
    print("\n[A] transformers 串行（一次一条）...")
    gen_a = 0
    t0 = time.time()
    for p in chat_prompts:
        enc = tokenizer(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen_a += int((out[0, enc["input_ids"].shape[1]:] != tokenizer.pad_token_id).sum())
    dt_a = time.time() - t0

    # [B] batch（decoder-only 必须左 padding）
    print(f"[B] transformers batch={batch_size}（手动 padding）...")
    tokenizer.padding_side = "left"
    gen_b = 0
    t0 = time.time()
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen_b += int((out[:, enc["input_ids"].shape[1]:] != tokenizer.pad_token_id).sum())
    dt_b = time.time() - t0

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "serial": {"time": dt_a, "gen_tokens": gen_a,
                   "qps": len(prompts) / dt_a, "tps": gen_a / dt_a},
        "batch":  {"time": dt_b, "gen_tokens": gen_b,
                   "qps": len(prompts) / dt_b, "tps": gen_b / dt_b},
    }


# ══════════════════════════════════════════════════════════════════
# 模式 C：vLLM（离线 API，内置 continuous batching）
# ══════════════════════════════════════════════════════════════════
def bench_vllm(model_path: str, prompts: list[str], max_new_tokens: int) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=2048,
              gpu_memory_utilization=0.9, dtype="float16", enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    chat_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True)
        for q in prompts
    ]

    print("[C] vLLM 批处理（内置 continuous batching）...")
    t0 = time.time()
    outputs = llm.generate(chat_prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens))
    dt_c = time.time() - t0
    gen_c = sum(len(o.outputs[0].token_ids) for o in outputs)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {"vllm": {"time": dt_c, "gen_tokens": gen_c,
                     "qps": len(prompts) / dt_c, "tps": gen_c / dt_c}}


# ── 绘图 + 保存 ────────────────────────────────────────────────
def plot_and_save(results: dict, out_dir: str, n_prompts: int, max_new_tokens: int):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "throughput_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_prompts": n_prompts, "max_new_tokens": max_new_tokens,
                   "batch_size": 8, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"JSON 结果保存：{json_path}")

    # 英文标签（避免 DejaVu Sans 缺中文字形）
    modes = ["transformers\nserial", "transformers\nbatch=8", "vLLM\ncontinuous\nbatching"]
    times = [results["serial"]["time"], results["batch"]["time"], results["vllm"]["time"]]
    qps = [results["serial"]["qps"], results["batch"]["qps"], results["vllm"]["qps"]]
    tps = [results["serial"]["tps"], results["batch"]["tps"], results["vllm"]["tps"]]
    colors = ["#aab7c4", "#82b1ff", "#69f0ae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], times, "Time (seconds)", f"Total Time for {n_prompts} Requests", "{:.1f}s"),
        (axes[1], qps, "QPS (requests/sec)", "Requests Per Second (higher is better)", "{:.1f}"),
        (axes[2], tps, "Tokens / sec", "Generation Throughput (tokens/sec)", "{:.0f}"),
    ]:
        bars = ax.bar(modes, vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom")
    plt.suptitle(f"vLLM vs Transformers: Throughput Benchmark (Qwen2-0.5B, {n_prompts} prompts)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "throughput_comparison.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"柱状图保存：{png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=50, help="prompt 数量（冒烟测试可改小）")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    prompts = build_prompts(args.n)
    print(f"Throughput Benchmark | {args.n} prompts × max {args.max_tokens} new tokens")
    print(f"模型: {args.model_path}")

    tf = bench_transformers(args.model_path, prompts, args.max_tokens, args.batch_size)
    vl = bench_vllm(args.model_path, prompts, args.max_tokens)
    results = {**tf, **vl}

    print("\n" + "=" * 78)
    print(f"{'模式':<28}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对vLLM'}")
    print("-" * 78)
    names = {"serial": "[A] transformers 串行",
             "batch": f"[B] transformers batch={args.batch_size}",
             "vllm": "[C] vLLM 批处理"}
    for k in ["serial", "batch", "vllm"]:
        r = results[k]
        print(f"{names[k]:<26}{r['time']:>7.2f}s   {r['qps']:>6.2f}    "
              f"{r['tps']:>7.0f}      {r['qps'] / results['vllm']['qps']:>5.2f}x")
    print("=" * 78)
    print("核心结论：")
    print(f"  vLLM 相对 transformers 串行加速：{results['vllm']['qps'] / results['serial']['qps']:.1f}x")
    print(f"  vLLM 相对 transformers batch:    {results['vllm']['qps'] / results['batch']['qps']:.1f}x")
    print("  提速来源：批处理收益 × (PagedAttention + continuous batching) 收益")

    plot_and_save(results, args.out_dir, args.n, args.max_tokens)


if __name__ == "__main__":
    main()
