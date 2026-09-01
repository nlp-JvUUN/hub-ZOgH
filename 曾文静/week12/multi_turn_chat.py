"""
multi_turn_chat.py — 多轮对话 REPL 入口

运行方式：
  # 离线验证（无需 API Key，mock 模拟器自带跨轮记忆）
  python multi_turn_chat.py --mock

  # 真实模型（需 DEEPSEEK_API_KEY，天气数据源 Open-Meteo 免费无 Key）
  python multi_turn_chat.py
  python multi_turn_chat.py --session a1b2c3d4      # 恢复上次会话
  python multi_turn_chat.py --transcript out.jsonl  # 全程轨迹落盘

  # 单轮脚本模式（借鉴洪建宇同学的双模式 CLI：给参数跑单轮，不给参数进 REPL）
  python multi_turn_chat.py --ask "宁德今天天气怎么样？" --session a1b2c3d4

对话命令（以 / 开头）：
  /new [标题]   开启新会话（标题默认取第一句问题）
  /list         列出所有会话（按最近活跃排序）
  /switch <id|序号>  切换会话（序号取 /list 展示的序号）
  /drop <id|序号>    删除会话
  /history [n]  回看当前会话最近 n 轮（默认全部）
  /summary      查看滚动摘要（超窗旧轮的压缩形态）
  /facts        查看已掌握的关键事实
  /stats        查看记忆统计（轮数/窗口/各块 token 估算）
  /reset        清空当前会话记忆（保留会话与历史文件）
  /help /exit   帮助 / 退出（exit quit 退出 均可）

与林书勤同学 week12（ChatAgent + REPL）的差异：
  他：messages 全量挂实例属性（含每轮工具中间过程），reset 只清内存；
  我：三层记忆（窗口/滚动摘要/关键事实）+ 会话级 JSONL 持久化，
      进程重启后 /switch 回来记忆仍在，/summary /facts 可检视记忆内部状态。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chat_agent import build_agent, ChatAgent  # noqa: E402
from session_store import SessionStore  # noqa: E402

COLORS = {
    "user": "\033[1;36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[1;93m",
    "dim": "\033[2m",
    "err": "\033[31m",
    "ok": "\033[1;92m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


class REPL:
    def __init__(self, agent: ChatAgent, store: SessionStore,
                 session_id: str | None = None, transcript=None):
        self.agent = agent
        self.store = store
        self.session_id = session_id or store.create()
        self.transcript = transcript
        self._restore()

    def _restore(self):
        """恢复会话：把 session 文件里的记忆快照灌回 MemoryManager。"""
        records = self.store.load_memory_records(self.session_id)
        if records["turns"]:
            self.agent.memory.load_records(records)
            s = self.store.get(self.session_id) or {}
            print(_c("ok", f"↩ 已恢复会话 {self.session_id}（{s.get('title','')}，"
                            f"历史 {len(records['turns'])} 轮）"))

    # ---- 会话管理命令 ----
    def _cmd_new(self, title: str = ""):
        self.session_id = self.store.create(title)
        self.agent.memory.reset()
        print(_c("ok", f"🔵 新会话 {self.session_id} 已开启"))

    def _cmd_list(self):
        sessions = self.store.list_sessions()
        if not sessions:
            print(_c("dim", "（暂无会话）"))
            return
        for i, s in enumerate(sessions, 1):
            mark = "▶" if s["session_id"] == self.session_id else " "
            print(f"{mark} [{i}] {s['session_id']}  {s['title']}  "
                  f"({s['turns']}轮, 更新于 {s['updated_at']})")

    def _resolve(self, key: str) -> str | None:
        sessions = self.store.list_sessions()
        if key.isdigit():
            i = int(key) - 1
            return sessions[i]["session_id"] if 0 <= i < len(sessions) else None
        return key if any(s["session_id"] == key for s in sessions) else None

    def _cmd_switch(self, key: str):
        sid = self._resolve(key)
        if sid is None:
            print(_c("err", f"找不到会话 {key}，用 /list 查看"))
            return
        # 记忆只在 chat() 期间变化，且每轮结束已随 turn 落盘，
        # 因此切换前无需额外保存，直接切换并恢复目标会话即可。
        self.session_id = sid
        self.agent.memory.reset()
        self._restore()

    def _cmd_drop(self, key: str):
        sid = self._resolve(key)
        if sid is None:
            print(_c("err", f"找不到会话 {key}"))
            return
        self.store.delete(sid)
        if sid == self.session_id:
            self.session_id = self.store.create()
            self.agent.memory.reset()
        print(_c("ok", f"🗑 已删除会话 {sid}"))

    # ---- 信息命令 ----
    def _cmd_history(self, n: int | None = None):
        turns = self.store.load_turns(self.session_id)
        if not turns:
            print(_c("dim", "（本会话还没有历史）"))
            return
        if n:
            turns = turns[-n:]
        for t in turns:
            print(_c("dim", f"── 第{t.get('turn')}轮 ──"))
            print(_c("user", f"问：{t.get('question')}"))
            print(f"答：{(t.get('answer') or '')[:200]}")

    def _cmd_summary(self):
        m = self.agent.memory
        print(_c("final", "📌 滚动摘要："))
        print(m.summary or _c("dim", "（暂无，窗口未溢出或摘要器未触发）"))

    def _cmd_facts(self):
        m = self.agent.memory
        print(_c("final", "🧠 关键事实："))
        for f in m.facts:
            print(f"  · {f}")
        if not m.facts:
            print(_c("dim", "（暂无）"))

    def _cmd_stats(self):
        m = self.agent.memory
        s = m.stats()
        print(f"记忆统计：累计轮次 {s['turns']}（窗口 {s['window']}）| "
              f"事实 {s['facts']} 条 | 摘要约 {s['summary_tokens']} tok | "
              f"上下文合计约 {s['context_tokens']} tok")

    # ---- 一轮对话 ----
    def chat_once(self, question: str):
        start_turn = self.agent.memory.stats()["turns"]
        for step in self.agent.chat(question):
            stype = step["type"]
            if stype == "action":
                print(f"\n  {_c('action', '🔧 ' + step['action'])} "
                      f"{json.dumps(step['action_input'], ensure_ascii=False)}")
                print(f"  {_c('obs', '👁 ' + step['observation'][:300])}")
            elif stype == "final":
                print(f"\n{_c('final', '✅ ' + step['answer'])}")
                u = step["usage"]
                print(_c("dim",
                         f"  （第{step['turn']['turn']}轮 · {step['terminated_by']} · "
                         f"工具 {len(step['turn']['tools'])} 次 · "
                         f"token {u['total_tokens']} · {step['elapsed']:.1f}s · "
                         f"记忆 {step['memory']['turns']} 轮/摘要{step['memory']['summary_tokens']}tok/"
                         f"事实{step['memory']['facts']}条）"))
            elif stype in ("max_steps", "dead_loop"):
                print(_c("err", f"⚠️  {step['answer']}"))

        # 落盘：本轮 turn 记录 + 当时记忆快照（summary/facts 随行保存，供恢复）
        # 只有正常产出 final 才落盘（异常中断时历史文件保持上次完整状态）
        if step.get("type") == "final":
            turn_rec = dict(step.get("turn", {}))
            turn_rec["summary"] = self.agent.memory.summary
            turn_rec["facts"] = self.agent.memory.facts
            self.store.save_turn(self.session_id, turn_rec)
        if self.transcript:
            self.transcript.write(
                json.dumps({"session_id": self.session_id, "question": question,
                            "final": step.get("answer", ""), "step": step},
                           ensure_ascii=False) + "\n")
            self.transcript.flush()


def main():
    ap = argparse.ArgumentParser(description="多轮对话版天气 Agent（week12 作业）")
    ap.add_argument("--mock", action="store_true", help="mock 模拟驱动，无需 API Key")
    ap.add_argument("--mock-tools", dest="mock_tools", action="store_true", default=None,
                    help="强制离线工具后端（默认：mock 模式离线，真实模式走 Open-Meteo）")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "dashscope"])
    ap.add_argument("--model", default="", help="覆盖默认模型名")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--window", type=int, default=6, help="短期记忆窗口轮数")
    ap.add_argument("--budget", type=int, default=4000, help="记忆 token 预算")
    ap.add_argument("--session", default=None, help="恢复指定会话")
    ap.add_argument("--sessions-dir", default=str(Path(__file__).parent / "sessions"))
    ap.add_argument("--transcript", default=None, help="全程轨迹导出 JSONL")
    ap.add_argument("--ask", default=None,
                    help="单轮模式：只回答这一个问题后退出（不传则进入交互 REPL）")
    args = ap.parse_args()

    use_mock_tools = args.mock_tools if args.mock_tools is not None else args.mock
    agent = build_agent(provider=args.provider, model=args.model, mock=args.mock,
                        max_steps=args.max_steps, window_turns=args.window,
                        token_budget=args.budget, mock_tools=use_mock_tools)
    store = SessionStore(Path(args.sessions_dir))
    fp = open(args.transcript, "w", encoding="utf-8") if args.transcript else None

    repl = REPL(agent, store, session_id=args.session, transcript=fp)
    mode = "mock（离线）" if args.mock else f"{args.provider}（真实模型）"

    # ── 单轮模式（--ask）：问答一次即退出，可配合 --session 续接历史 ──
    if args.ask:
        print(f"单轮模式 · {mode} · 会话 {repl.session_id}")
        repl.chat_once(args.ask)
        if fp:
            fp.close()
        return

    print("=" * 64)
    print(f"天气多轮对话 Agent · 模式: {mode} · 窗口 {args.window} 轮 · 预算 {args.budget} tok")
    print(f"当前会话: {repl.session_id}  （/help 查看命令）")
    print("=" * 64)

    try:
        while True:
            try:
                text = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not text:
                continue
            low = text.lower()
            if low in ("exit", "quit", "退出"):
                print("再见！")
                break
            if text.startswith("/"):
                parts = text.split(maxsplit=1)
                cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
                if cmd in ("/exit", "/quit"):
                    print("再见！")
                    break
                if cmd == "/new":
                    repl._cmd_new(rest)
                elif cmd == "/list":
                    repl._cmd_list()
                elif cmd == "/switch":
                    repl._cmd_switch(rest)
                elif cmd == "/drop":
                    repl._cmd_drop(rest)
                elif cmd == "/history":
                    repl._cmd_history(int(rest) if rest.isdigit() else None)
                elif cmd == "/summary":
                    repl._cmd_summary()
                elif cmd == "/facts":
                    repl._cmd_facts()
                elif cmd == "/stats":
                    repl._cmd_stats()
                elif cmd == "/reset":
                    repl.agent.memory.reset()
                    print(_c("ok", "🔁 当前会话记忆已清空"))
                elif cmd == "/help":
                    print(__doc__.split("运行方式")[0])
                    print("对话命令：/new /list /switch /drop /history /summary "
                          "/facts /stats /reset /help /exit")
                else:
                    print(_c("err", f"未知命令 {cmd}，/help 查看"))
            else:
                repl.chat_once(text)
    finally:
        if fp:
            fp.close()


if __name__ == "__main__":
    main()
