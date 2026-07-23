"""
吞吐对比 D：vLLM Server + OpenAI 客户端（HTTP 并发）

对比 bench_throughput.py 的 [A][B][C] 三路，新增 [D] Server 路径：
  - [A] transformers 串行       0.28 QPS
  - [B] transformers batch=8    1.52 QPS
  - [C] vLLM 离线批处理          3.38 QPS
  - [D] vLLM Server + HTTP 并发  ?.?? QPS  ← 这个脚本

仅跑实验，不生成图表。
图表由 plot_results.py 独立生成（加载已有 JSON，不用重跑 50 次）。

使用方式（需先启动 vLLM Server）：
  # 终端 1：bash start_server.sh
  # 终端 2：python bench_server.py           # 默认 5 并发
  #        python bench_server.py -c 10      # 调高到 10 并发
  #        python bench_server.py -c 1       # 模拟串行
  # 终端 3：python plot_results.py           # 单独出图（无需 Server）
"""

import argparse
import json
import os
import time
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
# 注册 Windows 本地中文字体（直接加载文件，绕过 findfont 的缓存问题）
_cjk_font = None
for _fpath in [
    r"C:\WINDOWS\Fonts\simhei.ttf",       # 黑体
    r"C:\WINDOWS\Fonts\msyh.ttc",          # 微软雅黑
    r"C:\WINDOWS\Fonts\NotoSansSC-VF.ttf", # Noto Sans SC
]:
    if os.path.exists(_fpath):
        try:
            font_manager.fontManager.addfont(_fpath)
            _cjk_font = font_manager.FontProperties(fname=_fpath).get_name()
            print(f"[Font] Loaded: {_cjk_font} <- {_fpath}")
            break
        except Exception as e:
            print(f"[Font] Failed to load {_fpath}: {e}")
if _cjk_font:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["axes.unicode_minus"] = False

# ── 配置（与 bench_throughput.py 保持一致）─────────────────────────────
MODEL = "qwen2.5-0.5b"
N_PROMPTS = 50
MAX_NEW_TOKENS = 100
CONCURRENCY = 5  # 默认并发数

# ── 同一套测试 prompts ────────────────────────────────────────────────
SHORT_QUESTIONS = [
    "什么是股票？", "什么是基金？", "什么是ETF？", "什么是债券？", "什么是期权？",
    "什么是熊市？", "什么是牛市？", "什么是PE？", "什么是ROE？", "什么是毛利率？",
]
MEDIUM_QUESTIONS = [
    "解释一下价值投资和趋势投资的区别。",
    "什么情况下应该止损？",
    "为什么会出现股市崩盘？",
    "沪深300和中证500有什么区别？",
    "什么是量化交易？",
    "基金定投的优势是什么？",
    "股票回购对股价有什么影响？",
    "可转债有哪些特点？",
    "如何判断一家公司是否值得投资？",
    "什么是做市商制度？",
]
LONG_QUESTIONS = [
    "请详细介绍一下巴菲特的投资理念及其核心原则，并举例说明。",
    "解释下现金流折现（DCF）估值法的基本步骤、使用的参数以及它的局限性。",
    "比较A股和美股在交易制度、监管环境、投资者结构等方面的主要差异。",
    "什么是技术分析？它和基本面分析有什么区别？两种方法各自的适用场景是什么？",
    "详细解释资产配置的核心思想，常见的几种配置模型，以及如何根据个人风险偏好调整。",
]
PROMPTS = (SHORT_QUESTIONS * 3 + MEDIUM_QUESTIONS * 1 + LONG_QUESTIONS * 2)[:N_PROMPTS]
assert len(PROMPTS) == N_PROMPTS


def bench_server(prompts: list[str], concurrency: int = 5) -> dict:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    n_total = len(prompts)
    total_gen_tokens = 0

    print(f"\n[D] vLLM Server + HTTP 并发（concurrency={concurrency}）...")
    t0 = time.time()

    def call(q: str) -> tuple[float, int]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
        content = resp.choices[0].message.content
        # 粗略估算生成的 token 数（用字符数近似，和 bench_throughput.py 的 token_ids 计数口径不同）
        gen_tokens = resp.usage.completion_tokens if resp.usage else len(content) // 2
        return gen_tokens

    # 用 ThreadPoolExecutor 并发
    import concurrent.futures
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call, q) for q in prompts]
        for f in concurrent.futures.as_completed(futures):
            gen_tokens = f.result()
            total_gen_tokens += gen_tokens
            done += 1
            if done % 10 == 0:
                print(f"    进度 {done}/{n_total}")

    dt = time.time() - t0
    qps = n_total / dt
    tps = total_gen_tokens / dt

    print(f"\n  [D] 完成: {dt:.2f}s  |  QPS={qps:.2f}  |  tokens/s={tps:.0f}")
    return {"time": round(dt, 2), "qps": round(qps, 2), "tps": round(tps, 0)}


def load_prev_results() -> dict | None:
    """读取 bench_throughput.py 产出的 through_results.json，做对比"""
    prev_path = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "throughput_results.json"
    )
    if os.path.exists(prev_path):
        with open(prev_path, "r") as f:
            return json.load(f)
    return None


def print_comparison(result_d: dict, prev: dict | None) -> dict:
    print("\n" + "=" * 90)
    print("  结果汇总（含全部 4 路对比）")
    print("=" * 90)
    print(f"{'模式':<30}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对vLLM':<10}")
    print("-" * 90)

    rows = []
    if prev:
        vllm_qps = prev["results"]["vllm"]["qps"]
        for k, label in [("serial", "[A] transformers 串行"),
                         ("batch",  "[B] transformers batch=8"),
                         ("vllm",   "[C] vLLM 离线批处理")]:
            r = prev["results"][k]
            rel = r["qps"] / vllm_qps
            rows.append((f"{label:<28}", r["time"], "s", r["qps"], r["tps"], rel))
            print(f"{label:<28}{r['time']:>6.2f}s     {r['qps']:>5.2f}     {r['tps']:>6.0f}      {rel:>5.2f}×")

    # [D] 当前 Server 结果
    d_qps = result_d["qps"]
    d_vllm_qps = prev["results"]["vllm"]["qps"] if prev else d_qps
    rel_d = d_qps / d_vllm_qps
    rows.append((f"[D] vLLM Server HTTP 并发", result_d["time"], "s", d_qps, result_d["tps"], rel_d))
    print(f"{'[D] vLLM Server HTTP 并发':<28}{result_d['time']:>6.2f}s     "
          f"{d_qps:>5.2f}     {result_d['tps']:>6.0f}      {rel_d:>5.2f}×")

    print("\n" + "=" * 90)
    print("  结论：Server 路径（HTTP + 网络序列化）比离线 vLLM 慢多少？")
    if prev:
        overhead = prev["results"]["vllm"]["qps"] / d_qps if d_qps > 0 else float("inf")
        print(f"    离线 vLLM → Server HTTP：{overhead:.2f}× 开销")
    print("=" * 90)

    # 返回合并结果供绘图
    combined = {}
    if prev:
        combined = {
            "n_prompts": prev["n_prompts"],
            "serial":  prev["results"]["serial"],
            "batch":   prev["results"]["batch"],
            "vllm":    prev["results"]["vllm"],
        }
    combined["server"] = {
        "time": result_d["time"],
        "qps":  result_d["qps"],
        "tps":  result_d["tps"],
    }
    combined["n_prompts"] = combined.get("n_prompts", 50)
    return combined


def plot_4way(results: dict, out_path: str):
    """绘制 4 路吞吐对比柱状图（含 [D] Server 路径）"""
    modes = ["transformers\nserial", "transformers\nbatch=8", "vLLM Server\nHTTP 并发", "vLLM\ncontinuous\nbatching"]
    times = [results["serial"]["time"], results["batch"]["time"],
             results["server"]["time"], results["vllm"]["time"]]
    qps = [results["serial"]["qps"], results["batch"]["qps"],
           results["server"]["qps"], results["vllm"]["qps"]]
    tps = [results["serial"]["tps"], results["batch"]["tps"],
           results["server"]["tps"], results["vllm"]["tps"]]
    colors = ["#aab7c4", "#82b1ff", "#ffb74d", "#69f0ae"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 总耗时
    bars = axes[0].bar(modes, times, color=colors)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Total Time for {results['n_prompts']} Requests")
    for b, v in zip(bars, times):
        axes[0].text(b.get_x() + b.get_width()/2, v, f"{v:.1f}s",
                     ha="center", va="bottom")

    # 2. QPS
    bars = axes[1].bar(modes, qps, color=colors)
    axes[1].set_ylabel("QPS (requests/sec)")
    axes[1].set_title("Requests Per Second (higher is better)")
    for b, v in zip(bars, qps):
        axes[1].text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                     ha="center", va="bottom")

    # 3. tokens/s
    bars = axes[2].bar(modes, tps, color=colors)
    axes[2].set_ylabel("Tokens / sec (generated)")
    axes[2].set_title("Generation Throughput (tokens/sec)")
    for b, v in zip(bars, tps):
        axes[2].text(b.get_x() + b.get_width()/2, v, f"{v:.0f}",
                     ha="center", va="bottom")

    plt.suptitle("vLLM vs Transformers: Throughput Benchmark (Qwen2.5-0.5B, GTX 1660 Ti 6GB)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n柱状图已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="vLLM Server 吞吐测试")
    parser.add_argument("-c", "--concurrency", type=int, default=5,
                        help="并发数（默认 5）")
    args = parser.parse_args()
    concurrency = args.concurrency

    print("=" * 70)
    print(f"  Throughput Benchmark — [D] Server Path")
    print(f"  {N_PROMPTS} prompts × max {MAX_NEW_TOKENS} tokens × concurrency={concurrency}")
    print("=" * 70)
    print("  前提：vLLM Server 已在 http://localhost:8000 运行")

    result_d = bench_server(PROMPTS, concurrency)

    # 保存（按并发数分别保存，不覆盖）
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bench_server_results_c{concurrency}.json")
    save = {
        "n_prompts": N_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "concurrency": concurrency,
        "result": result_d,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")

    print(f"\n实验完成。运行 python plot_results.py 查看 4 路对比图。")


if __name__ == "__main__":
    main()
