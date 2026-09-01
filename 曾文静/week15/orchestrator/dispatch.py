"""
dispatch.py — 并行派发引擎（Orchestrator-Workers 拓扑的 fan-out / fan-in）

教学重点（对应课件 Part 6 Graph Engineering）：
  1. fan-out：主 agent 一条 dispatch_workers 调用 → N 个 worker 并行执行
     （ThreadPoolExecutor，墙钟 ≈ max(各 worker 时长)，而非 sum）；
  2. fan-in：worker 结果以「结构化契约」回收（schema-first 交接，PPT 落地要点：
     每条边有结构化数据契约，下游不靠猜）——每个 worker 返回
     {node_id, skill, task, status, final_answer, trace, duration}，
     汇总时再格式化成主 agent 的 Observation 文本；
  3. 可观测性：graph_id / node_id 贯穿每个 worker 的 trace；
  4. serial 模式：同一批任务退化为 for 循环执行 —— 并行 vs 串行 A/B 的基线。

派发参数格式（主 agent 输出，管道分隔，每段 = 技能名: 任务）：
    weather: 北京 | weather: 上海 | file: 用中文总结 samples/notes_rag.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from react_loop import ReActLoop
from skills import SKILL_REGISTRY, list_skills_desc, resolve_skill

# 单次派发最多 worker 数（保护：防主 agent 一次派几十个）
MAX_WORKERS_PER_DISPATCH = 8


def parse_spec(action_input: str) -> list[dict]:
    """把派发参数解析成结构化任务契约 [{skill, task}]。
    容错：按 | 分段 → 每段按第一个 : 拆「技能: 任务」；
    技能名用 resolve_skill 解析（支持中文别名/模糊前缀）；
    解析不了的分段返回 error 标记，由主 agent 自行修正重试。"""
    tasks = []
    for seg in [s.strip() for s in action_input.split("|") if s.strip()]:
        if ":" in seg:
            name, _, task = seg.partition(":")
            key = resolve_skill(name.strip())
            if key is None:
                tasks.append({"error": f"未知技能「{name.strip()}」，可选技能见 list_skills"})
            elif not task.strip():
                tasks.append({"error": f"技能「{key}」缺少任务参数"})
            else:
                tasks.append({"skill": key, "task": task.strip()})
        else:
            tasks.append({"error": f"分段「{seg}」缺少技能名，格式应为 技能名: 任务"})
    return tasks


def _build_worker(task: dict, graph_id: str, node_id: str) -> ReActLoop:
    """按技能契约构造 worker（ReAct 循环 + 该技能的工具集与提示词）。"""
    spec = SKILL_REGISTRY[task["skill"]]
    return ReActLoop(
        agent_name=task["skill"],
        tools=spec["tools"],
        max_steps=spec.get("max_steps", 5),
        model_tag=f"deepseek-chat({task['skill']})",
        system_prompt=spec["prompt"],
        graph_id=graph_id,
        node_id=node_id,
    )


def dispatch_workers(action_input: str,
                     shared_state: Optional[dict] = None,
                     on_worker_step: Optional[Callable] = None,
                     on_worker_done: Optional[Callable] = None,
                     on_dispatch: Optional[Callable] = None,
                     serial: bool = False) -> str:
    """
    dispatch_workers 工具实现。
    - action_input: "技能名: 任务 | 技能名: 任务 | ..."（管道分隔）
    - 解析 → 构造 N 个 worker → serial=False 并行 / serial=True 串行执行
    - 收齐后：更新 shared_state（workers/dispatches/parallel_stats），
      返回汇总文本（每个 worker 结果截短）作为主 agent 的 Observation。
    """
    raw_tasks = parse_spec(action_input)[:MAX_WORKERS_PER_DISPATCH]
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("workers", {})
    shared_state.setdefault("dispatches", [])
    shared_state.setdefault("parallel_stats", [])

    # ── 契约校验：解析失败的段直接作为错误结果返回，不浪费 LLM 调用 ──
    if any("error" in t for t in raw_tasks):
        errs = [t["error"] for t in raw_tasks if "error" in t]
        return "派发参数解析失败：\n- " + "\n- ".join(errs) + \
               "\n请修正后用 dispatch_workers 重新派发。"

    graph_id = shared_state.get("graph_id", "g_" + uuid.uuid4().hex[:6])
    defs = []
    for task in raw_tasks:
        node_id = f"w_{uuid.uuid4().hex[:6]}"
        worker = _build_worker(task, graph_id, node_id)
        defs.append({"node_id": node_id, "worker": worker, "task": task})

    # ── fan-out 事件（拓扑可视化/日志用）───────────────────────────────
    dispatch_info = {"tasks": [{"skill": t["skill"], "task": t["task"]} for t in raw_tasks],
                     "worker_ids": [d["node_id"] for d in defs]}
    shared_state["dispatches"].append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    def _run_one(d: dict):
        node_id, worker, task = d["node_id"], d["worker"], d["task"]
        res = worker.run(task["task"], on_step=(
            lambda step, nid=node_id: on_worker_step(nid, step) if on_worker_step else None))
        result = {
            "node_id": node_id, "skill": task["skill"], "task": task["task"],
            "status": "ok" if not res["forced"] else "forced",
            "final_answer": res["final_answer"], "trace": res["trace"],
            "duration": res["duration"],
        }
        shared_state["workers"][node_id] = result
        if on_worker_done:
            on_worker_done(node_id, result["duration"], task["skill"], task["task"])
        return result

    # ── 执行：并行（ThreadPool）/ 串行（for 循环）───────────────────────
    t0 = time.time()
    results = []
    if serial:
        for d in defs:
            results.append(_run_one(d))
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, d): d for d in defs}
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: [d["node_id"] for d in defs].index(r["node_id"]))

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for r in results), 2)
    stats = {"n_workers": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
             "speedup": round(serial_sum / wall, 2) if wall else 0}
    shared_state["parallel_stats"].append(stats)

    # ── fan-in：汇总成主 agent 的 Observation（每个结果截短防 context 爆炸）──
    parts = [f"【worker {r['node_id']} | {r['skill']} | {r['task'][:40]}】"
             f"（用时 {r['duration']}s）\n{r['final_answer'][:600]}"
             for r in results]
    return (f"并行派发完成：{len(defs)} 个 worker，wall-clock {wall}s "
            f"（若串行需 {serial_sum}s，并行加速 {stats['speedup']}×）\n\n"
            + "\n\n".join(parts))


def _execute(defs: list[dict], serial: bool, runner: Callable) -> tuple[list, dict]:
    """底层执行器（与工具解耦，便于单测直接验证并行/串行差异）。"""
    t0 = time.time()
    results = []
    if serial:
        for d in defs:
            results.append(runner(d))
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(runner, d): d for d in defs}
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: [d["node_id"] for d in defs].index(r["node_id"]))
    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for r in results), 2)
    return results, {"wall_clock": wall, "serial_sum": serial_sum,
                     "speedup": round(serial_sum / wall, 2) if wall else 0}


if __name__ == "__main__":
    # 自测：解析器 + 技能注册表
    print("parse_spec 测试:")
    for spec in ["weather: 北京 | weather: 上海 | file: 总结 samples/notes_rag.md",
                 "天气: 广州 | 未知技能: xxx",
                 "没有技能名的分段"]:
        print(f"  {spec!r} → {parse_spec(spec)}")
    print("\n技能清单:\n" + list_skills_desc())
