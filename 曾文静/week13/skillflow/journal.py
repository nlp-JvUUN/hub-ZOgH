"""
记忆日志层 — 执行痕迹落成 Markdown 日记 + Memory Flush。

课件要点（slide 18-22）：
  - Markdown 是「人机双读」的记忆载体：人类直接可读可改，无需任何工具；
  - Memory Flush：把一次会话/一天的关键信息提炼后写入 MEMORY.md（会议纪要），
    对话历史本身（录音）留在每日日志里 —— 两者互补。

这里把同一思想用在 harness 自身：
  - 每个事件实时追加到 journal/YYYY-MM-DD.md（每日录音）与 events.jsonl；
  - flush() 把当天日志提炼成结构化摘要写进 journal/MEMORY.md（会议纪要），
    由 daily-report 心跳技能定期触发（Memory Flush 的"心跳任务触发"时机）。
  - 参考作业用 SQLite 存执行历史；这里刻意用纯文本 —— 打开目录就能看懂，
    也能直接 git 提交、diff。

说明： memory_flush 用 LLM 做"提炼"，因为要理解语义；
这里的事件都是结构化数据，用确定性规则摘要即可（零依赖、可复现），
接口留了 summarizer 参数，想上 LLM 随时可替换。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .model import Event


def _ev_line(event: Event) -> str:
    """把事件压成一行人类可读文本。"""
    payload = event.payload
    brief = ""
    if isinstance(payload, dict):
        brief = json.dumps(payload, ensure_ascii=False)[:160]
    elif payload is not None:
        brief = str(payload)[:160]
    ts = datetime.fromtimestamp(event.ts).strftime("%H:%M:%S")
    return f"- {ts} [{event.kind}] session={event.session} skill={event.skill} {brief}".rstrip()


class Journal:
    """Markdown 每日日志 + JSONL 事件流 + Memory Flush。"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.md_dir = self.base_dir / "md"
        self.jsonl_dir = self.base_dir / "jsonl"
        self.memory_file = self.base_dir / "MEMORY.md"
        for d in (self.md_dir, self.jsonl_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── 写入（每次事件实时落盘） ─────────────────────────────

    def log_event(self, event: Event):
        day = date.fromtimestamp(event.ts)
        md = self.md_dir / f"{day.isoformat()}.md"
        with md.open("a", encoding="utf-8") as f:
            f.write(_ev_line(event) + "\n")
        with (self.jsonl_dir / f"{day.isoformat()}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read_day(self, day: Optional[date] = None) -> str:
        day = day or date.today()
        p = self.md_dir / f"{day.isoformat()}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def list_days(self) -> List[str]:
        return sorted(p.stem for p in self.md_dir.glob("*.md"))

    # ── Memory Flush：当日日志 -> MEMORY.md 摘要 ──────────────

    def flush(
        self,
        day: Optional[date] = None,
        summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """
        把某一天（默认今天）的 jsonl 事件提炼成摘要，合并进 MEMORY.md。

        Returns: 生成的当天摘要文本。
        """
        day = day or date.today()
        records = self._read_jsonl(day)

        summary = summarizer(records) if summarizer else self._summarize(records)

        old = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
        old = old.strip()

        # 拆分头部（如 # MEMORY.md）与按天分节的内容
        head = "# MEMORY.md"
        rest = old
        if "## " in old:
            first = old.find("## ")
            head = old[:first].strip() or "# MEMORY.md"
            rest = old[first:]
            parts = rest.split("## ")
            kept = [p for p in parts if p and not p.startswith(day.isoformat())]
            rest = "## " + "## ".join(kept) if kept else ""

        section = f"## {day.isoformat()}\n{summary}\n"
        if rest:
            self.memory_file.write_text(head + "\n\n" + rest.rstrip() + "\n\n" + section, encoding="utf-8")
        else:
            self.memory_file.write_text(head + "\n\n" + section, encoding="utf-8")
        return summary

    def read_memory(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    # ── 内部 ─────────────────────────────────────────────────

    def _read_jsonl(self, day: date) -> List[Dict[str, Any]]:
        p = self.jsonl_dir / f"{day.isoformat()}.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    @staticmethod
    def _summarize(records: List[Dict[str, Any]]) -> str:
        """确定性规则摘要：按 skill × 状态聚合 + 少量结果样例。"""
        if not records:
            return "- 当日无执行事件\n"
        from collections import Counter

        skills: Dict[str, Counter] = {}
        samples: Dict[str, List[str]] = {}
        durations: Dict[str, List[float]] = {}
        for r in records:
            if r.get("skill", "").startswith("__"):
                continue
            name = r["skill"]
            skills.setdefault(name, Counter())
            samples.setdefault(name, [])
            durations.setdefault(name, [])
            kind = r["kind"]
            if kind.startswith("stage_"):
                skills[name][kind.replace("stage_", "")] += 1
            if kind == "stage_ok":
                dur = (r.get("payload") or {}).get("duration_ms")
                if dur is not None:
                    durations[name].append(dur)
                out = (r.get("payload") or {}).get("output")
                if isinstance(out, dict) and len(str(out)) < 200:
                    samples[name].append(json.dumps(out, ensure_ascii=False))

        lines = [f"- 执行流水共 {len(records)} 条事件"]
        for name in sorted(skills):
            c = skills[name]
            parts = [f"{k} {v}" for k, v in sorted(c.items())]
            dur = durations.get(name)
            dur_str = f"，平均耗时 {sum(dur)/len(dur):.0f}ms" if dur else ""
            lines.append(f"- {name}: {'、'.join(parts)}{dur_str}")
            for s in samples.get(name, [])[:2]:
                lines.append(f"  - 样例输出: {s}")
        return "\n".join(lines)
