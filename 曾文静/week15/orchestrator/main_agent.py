"""
main_agent.py — 主编排 Agent（Orchestrator-Workers 拓扑的 Supervisor）

教学重点（对应课件 Part 6 Graph Engineering）：
  1. 主 agent 自己是 ReAct 循环，但【不直接拥有任何业务工具】——
     它只有两个编排工具：
       - list_skills：查看可用 worker 技能清单（L1 元数据视图）
       - dispatch_workers：按「技能名: 任务 | 技能名: 任务」派发并行 worker
     —— 这是与"主 agent 直接拿业务工具干活"的架构的本质区别：
     业务能力全部下沉到 worker 技能注册表（skills.py），主 agent 只做
     拆解 → 派发 → 综合（纯 Orchestrator 职责，模型分层：主 agent 当"路由"）。
  2. LLM 自主路由：任务可拆成 2 个及以上独立子任务 → 必须派发并行；
     单一任务 → 派 1 个 worker 或直接回答。
  3. 拓扑运行时生长：派发几个 worker、用什么技能，都由主 agent 决定，
     不是固定拓扑（动态 Orchestrator-Workers，PPT 6.3）。
  4. 可观测性：整个运行生成一个 graph_id，主/worker 每条 trace 都带
     graph_id + node_id（PPT 落地要点：节点级可观测、可审计）。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import uuid
from typing import Callable, Optional

from dispatch import dispatch_workers
from react_loop import ReActLoop
from skills import list_skills_desc

MAIN_SYSTEM = """你是任务编排主 Agent（Orchestrator）。你不亲自执行业务任务，
而是把任务拆解后派发给并行 worker 完成，最后综合成最终答案。

你有 2 个编排工具：
- list_skills：查看当前可派发的 worker 技能清单（技能名、别名、参数要求）
- dispatch_workers：派发多个 worker 并行执行
  参数格式（管道分隔，每段 = 技能名: 任务）：
  "weather: 北京 | weather: 上海 | file: 用中文总结 samples/notes_rag.md"

【关键决策原则】
- 只要任务能拆成 2 个及以上独立子任务（多个城市、多个文件、多个调研对象等），
  必须用 dispatch_workers 并行派发，不要自己直接作答。
  示例：Question: 对比北京、上海、广州的天气
        Thought: 三个城市相互独立，应并行派发三个天气 worker
        Action: dispatch_workers
        Action Input: weather: 北京 | weather: 上海 | weather: 广州
- 单一子任务（1 个城市、1 个文件）也走 dispatch_workers 派 1 个 worker，
  让 worker 用专用技能处理，保证口径统一。
- 拿不到具体技能时先用 list_skills 查看清单，再决定派发方案。

【综合要求】
- 拿到各 worker 结果后，综合成结构化最终答案（分对象/分维度组织，带关键数据）；
- 末尾附一句并行执行统计（worker 数、wall-clock、串行估算、加速比）；
- 报告中的事实必须来自 Observation，不得编造。"""


def run(question: str, serial: bool = False,
        on_main_step: Optional[Callable] = None,
        on_worker_step: Optional[Callable] = None,
        on_worker_done: Optional[Callable] = None,
        on_dispatch: Optional[Callable] = None) -> dict:
    """
    执行一次编排任务。
    serial=True 时 worker 串行执行（parallel vs serial A/B 基线）。
    返回 {final_answer, main_trace, workers, dispatches, parallel_stats, graph_id, wall}。
    """
    graph_id = "g_" + uuid.uuid4().hex[:6]
    shared_state = {"graph_id": graph_id, "workers": {}, "dispatches": [],
                    "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        return dispatch_workers(
            action_input, shared_state=shared_state,
            on_worker_step=on_worker_step, on_worker_done=on_worker_done,
            on_dispatch=on_dispatch, serial=serial)

    t0 = time.time()
    main = ReActLoop(
        agent_name="main",
        tools={
            "list_skills": (lambda _: list_skills_desc(),
                            "查看可派发的 worker 技能清单，无需参数"),
            "dispatch_workers": (dispatch_tool,
                                 "派发多个 worker 并行执行，参数格式: 技能名: 任务 | 技能名: 任务"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
        graph_id=graph_id,
        node_id="main",
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "workers": shared_state["workers"],
        "dispatches": shared_state["dispatches"],
        "parallel_stats": shared_state["parallel_stats"],
        "graph_id": graph_id,
        "wall": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    q = "对比北京、上海、广州、深圳明天的天气情况，并给出出行建议"
    print(f"问题: {q}\n")
    r = run(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发 {len(r['dispatches'])} 次 | worker 数: {len(r['workers'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n最终报告（前 400 字）:\n{r['final_answer'][:400]}")
