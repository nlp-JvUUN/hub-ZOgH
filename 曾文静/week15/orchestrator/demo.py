"""
demo.py — 命令行演示：跑一次编排任务，输出拓扑图 + 节点 trace + 并行统计

用法：
  python -m orchestrator.demo                         # 默认天气对比演示
  python -m orchestrator.demo "你的问题"              # 自定义问题
  python -m orchestrator.demo "你的问题" --serial     # 串行模式（A/B 对比基线）
  python -m orchestrator.demo --question-file          # 演示多文件加工场景
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from .main_agent import run as _run  # noqa: E402

DEFAULT_Q = "对比北京、上海、广州、深圳明天的天气情况，并给出出行建议"
FILE_Q = ("处理 samples 目录下的三份笔记（notes_rag.md、notes_agent.md、notes_graph.md），"
          "分别用中文总结并提炼 3 个核心要点")

PALETTE = {"main": "\033[36m", "weather": "\033[32m", "file": "\033[33m", "reset": "\033[0m"}


def color(node_id: str, text: str) -> str:
    kind = node_id.split("_")[0] if node_id != "main" else "main"
    return f"{PALETTE.get(kind, '')}{text}{PALETTE['reset']}"


def print_topology(r: dict) -> None:
    print(f"\n{'─'*56}\n🌐 拓扑图  graph={r['graph_id']}  （总墙钟 {r['wall']}s）\n{'─'*56}")
    print(f"{color('main', '● main')} 主编排 Agent")
    for disp in r["dispatches"]:
        for w in disp["worker_ids"]:
            info = r["workers"].get(w, {})
            skill = info.get("skill", "?")
            dur = f"{info.get('duration', 0)}s" if info else "?"
            mark = "✓" if info.get("status") == "ok" else "!"
            print(f"  ├─ {color(w, f'● {w}')} [{skill}] {mark} {dur}")
    for i, st in enumerate(r["parallel_stats"]):
        print(f"  └─ 第{i+1}次派发: {st['n_workers']} workers | "
              f"并行 {st['wall_clock']}s vs 串行 {st['serial_sum']}s | "
              f"加速 {st['speedup']}×")
    print("─" * 56)


def print_trace(node_id: str, r: dict) -> None:
    """打印某个节点的完整 ReAct trace。"""
    nodes = {"main": r["main_trace"]}
    nodes.update({wid: w["trace"] for wid, w in r["workers"].items()})
    if node_id not in nodes:
        print(f"未知节点: {node_id}，可选: {', '.join(nodes)}")
        return
    print(f"\n📜 节点 {color(node_id, node_id)} ReAct trace:")
    for s in nodes[node_id]:
        print(f"  [{s['idx']}] Thought: {s['thought'][:90]}")
        if s["action"] == "Final Answer":
            print(f"      → Final Answer: {s['action_input'][:120]}...")
        else:
            print(f"      → Action: {s['action']} | Input: {s['action_input'][:60]}")
            obs = (s.get("observation") or "")[:110].replace("\n", " ")
            print(f"      → Observation: {obs}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrator 并行编排演示")
    ap.add_argument("question", nargs="?", default=None)
    ap.add_argument("--serial", action="store_true", help="串行模式（A/B 基线）")
    ap.add_argument("--question-file", action="store_true", help="演示多文件加工场景")
    ap.add_argument("--trace", default=None, help="打印指定节点 trace（node_id 或 main）")
    args = ap.parse_args()

    q = args.question or (FILE_Q if args.question_file else DEFAULT_Q)
    mode = "串行" if args.serial else "并行"
    print(f"❓ 问题: {q}\n  模式: {mode} 执行")

    r = _run(q, serial=args.serial)
    print_topology(r)

    if args.trace:
        print_trace(args.trace, r)

    print(f"\n📋 最终报告（前 1500 字）:\n{r['final_answer'][:1500]}")
    print(f"\n✅ 完成 | graph={r['graph_id']} | 总墙钟 {r['wall']}s")


if __name__ == "__main__":
    main()
