"""
react_loop.py — 通用 ReAct 循环引擎（主 agent 与 worker 共用）

教学重点（对应课件 Part 6 Graph Engineering 的 L4 Loop Engineering）：
  1. ReAct = Reason + Act：LLM 生成 Thought → Action（工具名）→ Action Input（参数），
     执行工具得到 Observation，喂回 LLM 继续，直到 Final Answer。
  2. 主 agent 与 worker 是同一个类，区别只在 tools 字典与 system_prompt ——
     这就是"多 agent 拓扑"里节点可复用的最小单元。
  3. 完整 trace 捕获：每步 Thought/Action/ActionInput/Observation 都记录，
     且带 graph_id / node_id（PPT 落地要点：节点级可观测，graph_id/run_id/node_id 审计）。

实现细节（与常规 ReAct 的差异点）：
  - 用 stop=["Observation:"] 让模型生成完 Action Input 就停，等工具结果再续写；
  - Action Input 支持跨行（直到下一个标记或文本结束），不怕参数里有换行；
  - 解析兜底：模型拿到长结果后常直接写报告、不带 Final Answer 前缀，
    检测到「无 Action 但有实质文本」时当作 Final Answer，避免空 action 死循环；
  - 工具失败不抛异常：任何异常都转成 [工具执行出错] 观察文本，让模型自我修正。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re
import time
from typing import Callable, Optional

from llm_client import llm_chat

# 默认 ReAct 系统提示（worker 用；主 agent 有自己专属的编排提示词）
REACT_SYSTEM = """你是一个任务执行 Agent。你可以使用以下工具完成任务：

{tools_desc}

输出格式（每轮严格按此格式，只输出一次动作）：
Thought: 你的推理，说明这一步要做什么、为什么
Action: 工具名
Action Input: 工具参数

工具执行后会返回 Observation。多轮调用直到信息足够，最后输出：
Thought: 我已收集足够信息
Final Answer: 最终结果

规则：
- Action 必须是上面列出的工具名之一，Action Input 是它的参数字符串
- 每轮只调用一次工具，等 Observation 后再决定下一步
- 严禁编造工具返回内容，一切以 Observation 为准"""


def build_tools_desc(tools: dict) -> str:
    """tools: {name: (fn, desc)} → 工具清单文本"""
    return "\n".join(f"- {name}: {desc}" for name, (fn, desc) in tools.items())


class ReActLoop:
    """通用 ReAct 循环。主 agent / 每个 worker 各自实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None,
                 graph_id: str = "", node_id: str = ""):
        """
        agent_name:  节点名（main / 技能名），用于 trace 与拓扑展示
        tools:       {工具名: (fn(arg)->str, 描述)}
        system_prompt: 自定义系统提示；None 用默认 REACT_SYSTEM（{tools_desc} 会被替换）
        graph_id/node_id: 可观测性字段（PPT 6.3 落地要点），贯穿 trace
        """
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.graph_id = graph_id
        self.node_id = node_id
        self.trace: list[dict] = []

    def run(self, question: str, on_step: Optional[Callable] = None,
            shared_state: Optional[dict] = None) -> dict:
        """
        执行 ReAct 循环。
        on_step(step): 每完成一步回调（流式输出/可视化用）。
        shared_state: 跨节点共享状态（主 agent 派发 worker 时 worker 结果也存这里）。
        返回 {final_answer, trace, duration, node_id}。
        """
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""
        forced = False

        for idx in range(self.max_steps):
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": idx, "agent": self.agent_name, "node_id": self.node_id,
                    "graph_id": self.graph_id, "thought": thought,
                    "action": action, "action_input": action_input,
                    "observation": None, "final": False}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            # 执行工具（可能很慢，如 dispatch_workers 要等所有 worker 跑完）
            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation
            self.trace.append(step)
            if on_step:
                on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # 超过 max_steps：强制收尾，不抛错
            forced = True
            final_answer = ("（已达最大步数强制收尾）" +
                            (self.trace[-1].get("observation", "") if self.trace else ""))
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "node_id": self.node_id, "graph_id": self.graph_id,
                    "thought": "达到步数上限，强制收尾", "action": "Final Answer",
                    "action_input": final_answer, "observation": None,
                    "final": True, "forced": True}
            self.trace.append(step)
            if on_step:
                on_step(step)

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2), "node_id": self.node_id,
                "forced": forced}

    # ── 解析 ────────────────────────────────────────────────────────────
    def _parse(self, text: str) -> tuple[str, str, str]:
        """从 LLM 输出解析 (thought, action, action_input)。
        Final Answer 时 action == 'Final Answer'，内容放 action_input。
        兜底：无任何格式标记但有实质文本 → 当作 Final Answer。"""
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer|Observation):|$)",
                      text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*?)(?=\n|$)", text)
        if ma:
            action = ma.group(1).strip()
            # Action Input 跨行捕获：直到下一个标记或文本结束
            mi = re.search(r"Action Input:\s*(.*?)(?=\nThought:|\nFinal Answer:|\nObservation:|\Z)",
                           text, re.S)
            action_input = (mi.group(1).strip() if mi else "")
            # 清理：模型偶尔在参数后补一句旁白（如"（等待工具结果）"），
            # 去掉末尾的括号式旁白行，避免污染工具参数
            lines = action_input.splitlines()
            while lines and re.match(r"^[（(]", lines[-1].strip()):
                lines.pop()
            action_input = "\n".join(lines).strip()
            return thought, action, action_input

        if text.strip():
            return (thought or "综合已有信息直接作答"), "Final Answer", text.strip()
        return thought, "", ""

    # ── 工具执行 ────────────────────────────────────────────────────────
    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        if action not in self.tools:
            return (f"工具 '{action}' 不存在，可选: {', '.join(self.tools)}。"
                    f"请重新选择。")
        fn, _ = self.tools[action]
        try:
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            return str(fn(action_input))
        except Exception as e:  # noqa: BLE001
            return f"[工具执行出错] {type(e).__name__}: {str(e)[:150]}"


if __name__ == "__main__":
    # 自测：无工具、纯 LLM 回答
    import logging
    logging.basicConfig(level=logging.WARNING)
    loop = ReActLoop("test", tools={}, max_steps=3)
    r = loop.run("用一句话介绍 ReAct 是什么？")
    print(f"\n答案: {r['final_answer'][:120]}")
    print(f"trace {len(r['trace'])} 步: {[s['action'] for s in r['trace']]}")
