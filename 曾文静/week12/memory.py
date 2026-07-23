"""
memory.py — 多轮对话的分层记忆（MemoryManager）

作业核心（对照老师课件 agent教学.pptx「记忆层」一节）：
  单轮 Agent 的记忆 = 0：messages 在函数内创建、调用完即销毁，第二轮一切重来。
  多轮 Agent 的记忆 = 分层状态：把「上一轮说了什么、查到了什么」变成跨轮可见的上下文。

  课件把记忆分成四类（短期 / 长期 / 情景 / 语义），本作业用三层落地：

  ┌──────────────────────────────────────────────────────────────┐
  │ ① 短期记忆 short-term window                                  │
  │    最近 window_turns 轮的 (问题, 回答) 原文保留，逐字进入上下文  │
  │    对应课件：In-Context Memory，受 Context Window 限制         │
  ├──────────────────────────────────────────────────────────────┤
  │ ② 滚动摘要 rolling summary（情景记忆的压缩形态）                 │
  │    超过窗口的旧轮不直接丢弃，而是被 LLM 压缩成一段「此前对话摘要」，│
  │    摘要本身再参与后续压缩（滚动），对话可以无限长而 token 不爆炸   │
  │    对应课件：情景记忆 Episodic Memory（具体事件序列的浓缩）       │
  ├──────────────────────────────────────────────────────────────┤
  │ ③ 关键事实 key facts（语义记忆的轻量形态）                       │
  │    每轮结束后从工具结果/回答中抽取结构化事实（城市→坐标/温度/AQI），│
  │    单独存放并注入上下文，追问时直接命中，不用重查工具             │
  │    对应课件：语义记忆 Semantic Memory（抽象事实与概念）           │
  └──────────────────────────────────────────────────────────────┘

与常见作业（把全部历史消息原样拼进 messages）的本质区别：
  - 工具调用中间过程（Thought/Action/Observation）不入长期记忆，
    每轮只沉淀「问题 + 最终回答 + 事实」三样东西 —— 上下文增长 O(轮数) 而非 O(token)；
  - 有明确的 token 预算与压缩触发器，而不是等 Context Window 爆了才处理；
  - 记忆的读写时机遵循课件：每轮开始读（组装上下文），每轮结束写（追加+抽事实+可能压缩）。

设计要点：
  - 本模块不依赖任何 LLM / 工具实现：摘要与事实抽取通过注入的
    summarizer / fact_extractor 回调完成（chat_agent.py 里注入真实 LLM 或 mock 规则），
    因此记忆层可平移到课件的金融 Agent 或任意工具集。
  - 每条记忆记录带 turn 编号与时间戳，可完整序列化为 dict，
    供 session_store.py 持久化后跨进程恢复。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ── Token 估算 ──────────────────────────────────────────────────────────────
# 中文约 1 token ≈ 1.5~2 字符，按 len(text) * 0.65 粗略估算即可满足预算控制；
# 真实计费以模型 usage 为准（chat_agent 会记录实际 token）。
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) * 0.65))


# ── 记忆中的一轮对话 ────────────────────────────────────────────────────────

@dataclass
class Turn:
    turn: int                 # 轮次编号（从 1 开始）
    question: str             # 用户问题（原文）
    answer: str               # 最终回答（原文）
    tools: List[str] = field(default_factory=list)   # 本轮用到的工具名（仅统计用）
    facts: List[str] = field(default_factory=list)   # 本轮抽取的新事实
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn, "question": self.question, "answer": self.answer,
            "tools": self.tools, "facts": self.facts, "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            turn=d.get("turn", 0), question=d.get("question", ""),
            answer=d.get("answer", ""), tools=d.get("tools", []),
            facts=d.get("facts", []), ts=d.get("ts", 0.0),
        )


# ── 分层记忆管理器 ───────────────────────────────────────────────────────────

class MemoryManager:
    """
    三层记忆的状态容器与读写逻辑。

    注入回调（均可为空，为空则跳过对应功能）：
      summarizer(summary: str, evicted: list[Turn]) -> str
          把「旧摘要 + 被挤出窗口的若干轮」压缩成新摘要（滚动压缩）。
      fact_extractor(turn: Turn, tool_log: list, known_facts: list) -> list[str]
          从本轮回答与工具结果中抽取关键事实（known_facts 用于坐标反查城市）。

    上下文组装（build_context）产物结构：
      messages = [
        {"role": "system", "content": 系统提示 + 摘要块 + 事实块},
        *最近窗口的 Q/A 对,
        {"role": "user", "content": 本轮问题},
      ]
    """

    def __init__(
        self,
        window_turns: int = 6,
        token_budget: int = 4000,
        summarizer: Optional[Callable[[str, List[Turn]], str]] = None,
        fact_extractor: Optional[Callable[[Turn, list], List[str]]] = None,
    ):
        self.window_turns = window_turns
        self.token_budget = token_budget
        self.summarizer = summarizer
        self.fact_extractor = fact_extractor

        self.summary: str = ""          # ② 滚动摘要（此前对话的浓缩）
        self.facts: List[str] = []      # ③ 关键事实（去重后的全局事实表）
        self.turns: List[Turn] = []     # ① 全部轮次（窗口取尾部，超窗的进摘要）
        self._turn_seq: int = 0         # 单调递增轮次编号（窗口压缩不影响编号）

    # ── 写：每轮结束调用 ─────────────────────────────────────────────────────
    def end_turn(self, question: str, answer: str, tool_log: Optional[list] = None) -> Turn:
        tool_log = tool_log or []
        self._turn_seq += 1
        turn = Turn(
            turn=self._turn_seq,
            question=question,
            answer=answer,
            tools=[t.get("name", "?") for t in tool_log],
        )
        # 事实抽取（真实 LLM 或 mock 规则，由调用方注入；传入已知事实供坐标反查城市）
        if self.fact_extractor is not None:
            try:
                turn.facts = self.fact_extractor(turn, tool_log, list(self.facts)) or []
            except Exception:
                turn.facts = []
        # 新事实并入全局事实表（去重、限量，防止事实表自身膨胀）
        for f in turn.facts:
            if f not in self.facts:
                self.facts.append(f)
        self.facts = self.facts[-50:]

        self.turns.append(turn)
        self._compact_if_needed()
        return turn

    # ── 读：每轮开始调用 ─────────────────────────────────────────────────────
    def build_context(self, question: str, system_prompt: str) -> list:
        """
        组装发给模型的 messages：
        system 提示词尾部附加「摘要块 + 事实块」，随后是窗口内 Q/A 对。
        """
        extra = []
        if self.summary:
            extra.append(f"[此前对话摘要] {self.summary}")
        if self.facts:
            fact_lines = "；".join(self.facts)
            extra.append(f"[已掌握的关键事实] {fact_lines}")
        sys_content = system_prompt
        if extra:
            sys_content += "\n\n" + "\n".join(extra)

        messages = [{"role": "system", "content": sys_content}]
        # 窗口内最近几轮：以 Q/A 对呈现（工具中间过程不进长期记忆）
        for t in self.turns[-self.window_turns:]:
            messages.append({"role": "user", "content": f"（第{t.turn}轮）{t.question}"})
            messages.append({"role": "assistant", "content": t.answer})
        messages.append({"role": "user", "content": question})
        return messages

    # ── 压缩：把超窗旧轮滚动进摘要 ────────────────────────────────────────────
    def _compact_if_needed(self) -> None:
        turns = self.turns

        # 1) 超窗的轮次 → 进摘要（滚动压缩：旧摘要 + 被挤出的轮次）
        evictable = turns[: max(0, len(turns) - self.window_turns)]
        if evictable and self.summarizer is not None:
            self.summary = self.summarizer(self.summary, list(evictable))
            del turns[: len(evictable)]
            # 摘要本身也会膨胀，超过预算时再次压缩到只剩最近一次摘要的骨架
            if estimate_tokens(self.summary) > self.token_budget // 2 and self.summarizer is not None:
                self.summary = self.summarizer(self.summary, [])

        # 2) token 预算兜底：即使窗口内轮次超预算，也从最旧的开始挤进摘要
        while turns and estimate_tokens(self._window_text(turns)) > self.token_budget:
            oldest = turns.pop(0)
            if self.summarizer is not None:
                self.summary = self.summarizer(self.summary, [oldest])
            else:
                # 没有摘要器时的兜底：保留该轮的关键事实，正文丢弃
                self.summary = (self.summary + f"；第{oldest.turn}轮：{oldest.question}→{oldest.answer[:60]}")
                self.summary = self.summary[-2000:]

    @staticmethod
    def _window_text(turns: List[Turn]) -> str:
        return "".join(f"Q:{t.question} A:{t.answer}" for t in turns)

    # ── 状态导出 / 恢复（供会话持久化） ───────────────────────────────────────
    def to_records(self) -> dict:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "turns": [t.to_dict() for t in self.turns],
        }

    def load_records(self, records: dict) -> None:
        self.summary = records.get("summary", "")
        self.facts = list(records.get("facts", []))
        self.turns = [Turn.from_dict(d) for d in records.get("turns", [])]
        # 恢复后按当前窗口/预算配置立即做一次一致性压缩，
        # 保证「恢复的会话」与「一直在线的会话」行为一致
        self._compact_if_needed()
        # 轮次编号从已恢复的最大编号继续（防止恢复后编号回绕）
        self._turn_seq = max([t.turn for t in self.turns] or [0])

    def reset(self) -> None:
        """清空记忆（保留窗口/预算配置），相当于"失忆重来"。"""
        self.summary = ""
        self.facts = []
        self.turns = []
        self._turn_seq = 0

    def stats(self) -> dict:
        return {
            "turns": len(self.turns),
            "window": self.window_turns,
            "summary_tokens": estimate_tokens(self.summary),
            "facts": len(self.facts),
            "facts_tokens": estimate_tokens("；".join(self.facts)),
            "context_tokens": estimate_tokens(
                self.summary + "；".join(self.facts) + self._window_text(self.turns)),
        }
