"""
eval_compare.py — Parallel vs Serial 量化对比（凸显 subagent 并行优势）

教学重点：
  同一组编排问题，worker 分别用「并行（ThreadPoolExecutor）」和
  「串行（for 循环）」两种方式执行，对比 wall-clock，量化并行加速。
  并行的意义不是少做事，而是把 N 个独立子任务的墙钟从 sum 压到 ≈max。

用法：
  python -m orchestrator.eval_compare            # 全部问题
  python -m orchestrator.eval_compare --limit 1  # 只跑第 1 题（快速版）
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.WARNING)

from .main_agent import run as _run  # noqa: E402

EVAL_TASKS = [
    "对比北京、上海、广州、深圳明天的天气情况，并给出出行建议",
    "处理 samples 目录下的三份笔记（notes_rag.md、notes_agent.md、notes_graph.md），"
    "分别用中文总结并提炼 3 个核心要点",
    "对比杭州、成都、重庆、西安、武汉五座城市的天气与出行建议",
]


def run_one(question: str, serial: bool) -> dict:
    t0 = time.time()
    r = _run(question, serial=serial)
    wall = round(time.time() - t0, 2)
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": wall,
        "n_workers": ps["n_workers"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
        "n_llm_calls": sum(len(w["trace"]) for w in r["workers"].values())
                       + len(r["main_trace"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    qs = EVAL_TASKS[:args.limit] if args.limit else EVAL_TASKS

    results = []
    for i, q in enumerate(qs):
        print(f"[{i+1}/{len(qs)}] {q[:40]}...")
        p = run_one(q, serial=False)
        s = run_one(q, serial=True)
        results.append({"question": q, "parallel": p, "serial": s})
        print(f"    并行 {p['wall']}s（dispatch {p['dispatch_wall']}s, "
              f"{p['n_workers']} workers, 加速 {p['speedup']}×）"
              f" vs 串行 {s['wall']}s")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)
    print(f"\n{'='*60}\nParallel vs Serial 对比（{len(results)} 题）\n{'='*60}")
    print(f"{'指标':<18}{'并行(ThreadPool)':<20}{'串行(for循环)':<20}")
    print(f"{'平均总墙钟(s)':<18}{avg_p:<20.2f}{avg_s:<20.2f}")
    print(f"{'平均派发加速':<18}{avg_spd:<20.2f}×")
    print(f"\n结论：N 个独立 worker 并行，墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×；总墙钟加速受主 agent 串行编排段限制"
          f"（Amdahl 定律：可并行部分才受益）。")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                       "avg_serial_s": round(avg_s, 2),
                       "avg_speedup": round(avg_spd, 2)},
           "details": results}
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "eval_compare.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
