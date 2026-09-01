"""
CLI / REPL 入口 — 与 HTTP 网关共享同一个 HarnessApp。

用法示例：
    python -m skillflow scan
    python -m skillflow info word-count
    python -m skillflow run word-count text="hello world"
    python -m skillflow pipe "fetch-source | word-count | format-report"
    python -m skillflow chat "统计这句话的单词数：hello world"
    python -m skillflow unload word-count
    python -m skillflow heartbeat --once
    python -m skillflow watch
    python -m skillflow serve --port 8620
    python -m skillflow repl
    python -m skillflow flush
    python -m skillflow journal
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .app import HarnessApp
from .model import Event

DEFAULT_SKILLS = Path(__file__).resolve().parent.parent / "skills"

_ICONS = {
    "stage_ok": "✅",
    "stage_fail": "❌",
    "stage_skip": "⏭️ ",
    "stage_defer": "💤",
    "progress": "▸",
    "load": "📦",
    "discover": "🔎",
    "report": "📋",
    "heartbeat": "💓",
}


def print_event(ev: Event, verbose: bool = False):
    icon = _ICONS.get(ev.kind, "·")
    if ev.kind == "progress":
        print(f"  {icon} {ev!r}")
    else:
        print(f"{icon} {ev!r}")
    if verbose and ev.kind == "report":
        payload = ev.payload
        for st in payload.get("stages", []):
            print(f"      - {st['skill']}: {st['status']} ({st['duration_ms']:.0f}ms)")


def _parse_inputs(pairs: List[str]) -> Dict[str, Any]:
    """把 k=v 或 k=json 解析成输入字典。"""
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"输入需要 k=v 格式: {pair!r}")
        k, _, v = pair.partition("=")
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def build_app(args: argparse.Namespace, auto_scan: bool = True) -> HarnessApp:
    skills_dir = Path(getattr(args, "skills_dir", None) or DEFAULT_SKILLS)
    return HarnessApp(
        skills_dir=skills_dir,
        budget=args.budget if hasattr(args, "budget") and args.budget else 100,
        auto_scan=auto_scan,
    )


def cmd_scan(args) -> int:
    app = build_app(args, auto_scan=False)  # 由 scan 命令自己完成首次扫描，展示真实增量
    result = app.scan(force=args.force)
    print(f"发现 {len(result['skills'])} 个 skills（变化 {len(result['changed'])} 个）")
    for spec_dict in result["skills"]:
        hb = f" 心跳={spec_dict['heartbeat']}" if spec_dict["heartbeat"] else ""
        deps = f" 依赖={spec_dict['deps']}" if spec_dict["deps"] else ""
        print(f"  - {spec_dict['name']} v{spec_dict['version']} (weight={spec_dict['weight']}){hb}{deps}")
        print(f"      {spec_dict['description']}")
    if result["deps_errors"]:
        for err in result["deps_errors"]:
            print(f"  ⚠ {err}")
    print(f"加载预算: {result['budget']}")
    return 0


def cmd_info(args) -> int:
    app = build_app(args)
    try:
        data = app.info(args.skill)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    print(f"# {data['name']} v{data['version']} (weight={data['weight']})")
    print(data["description"])
    if data["deps"]:
        print(f"依赖: {data['deps']}")
    if data["heartbeat"]:
        print(f"心跳: {data['heartbeat']}")
    print("输入契约:")
    for k, v in data["consumes"].items():
        req = "必填" if v.get("required") else "可选"
        print(f"  - {k} ({v.get('type', 'any')}) [{req}] {v.get('desc', '')}")
    print(f"输出契约: {data['provides']}")
    print(f"资源清单 (L3): {data['resources']}")
    print(f"实现已加载 (L2): {data['impl_loaded']}  ← 未执行前永远是 False（懒加载）")
    return 0


def cmd_run(args) -> int:
    app = build_app(args)
    inputs = _parse_inputs(args.input)
    content = {"skill": args.skill, "inputs": inputs, "config": {"on_failure": args.policy}}
    events = app.run_stream(args.session, content, on_event=lambda ev: print_event(ev, args.verbose))
    report = events[-1].payload if events else {"status": "failed"}
    print(f"\n状态: {report.get('status')} | {report.get('message', '')}")
    return 0 if report.get("status") in ("ok", "partial") else 1


def cmd_pipe(args) -> int:
    app = build_app(args)
    inputs = _parse_inputs(args.input)
    content = {"pipe": args.pipeline, "inputs": inputs, "config": {"on_failure": args.policy}}
    events = app.run_stream(args.session, content, on_event=lambda ev: print_event(ev, args.verbose))
    report = events[-1].payload if events else {"status": "failed"}
    print(f"\n状态: {report.get('status')} | {report.get('message', '')}")
    for st in report.get("stages", []):
        out = st.get("output")
        brief = json.dumps(out, ensure_ascii=False)[:120] if out is not None else ""
        print(f"  - {st['skill']}: {st['status']} {brief}")
    return 0 if report.get("status") in ("ok", "partial") else 1


def cmd_chat(args) -> int:
    """自然语言入口：由 agent-react 元技能（ReAct 循环）调度其他技能。"""
    app = build_app(args)
    inputs = {"question": args.question}
    if args.max_iterations:
        inputs["max_iterations"] = args.max_iterations

    answer = "（未产生回答）"
    for ev in app.run_stream(args.session, {"skill": "agent-react", "inputs": inputs}):
        if ev.kind == "progress":
            print(f"  ▸ {ev!r}")
        elif ev.kind == "stage_ok":
            out = ev.payload.get("output") or {}
            answer = out.get("answer", answer)
            steps = out.get("steps", [])
            for s in steps:
                if s["action"] == "call_tool":
                    print(f"  ⚙ 调用技能 {s['tool']}({json.dumps(s['params'], ensure_ascii=False)})")
                elif s["action"] == "observation":
                    mark = "✓" if s["success"] else "✗"
                    print(f"  {mark} 观察: {str(s['observation'])[:100]}")
        else:
            print_event(ev)
    print(f"\n💬 {answer}")
    return 0


def cmd_unload(args) -> int:
    app = build_app(args)
    before = app.runtime.is_loaded(args.skill)
    app.runtime.unload(args.skill)
    print(f"{args.skill}: {'已卸载（下次执行会重新加载）' if before else '本未加载，无需卸载'}")
    return 0


def cmd_heartbeat(args) -> int:
    app = build_app(args)
    if args.once:
        msgs = app.scheduler.run_due_now()
        names = [m.content.get("skill") for m in msgs]
        print(f"触发心跳技能: {names}")
        for m in msgs:
            ok = app.hub.wait_for(m, timeout=15)
            print(f"  - {m.content.get('skill')}: {'已完成' if ok else '等待超时'}")
        return 0
    print("心跳调度器运行中（Ctrl+C 退出）…")
    try:
        app.scheduler.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_watch(args) -> int:
    app = build_app(args)
    print(f"watch 模式：监听 {app.skills_dir}（Ctrl+C 退出）")
    print("  向该目录放入新 skill 目录，或修改已有 SKILL.md / skill.py，无需重启。")
    app.start_watch(interval=args.interval)
    try:
        app.scheduler.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop_watch()
    return 0


def cmd_serve(args) -> int:
    from .gateway import Gateway

    app = build_app(args)
    gateway = Gateway(app, host=args.host, port=args.port, watch=not args.no_watch)
    gateway.start()
    print(f"技能清单: {len(app.registry.list_all())} 个")
    gateway.serve_forever()
    return 0


def cmd_flush(args) -> int:
    app = build_app(args)
    summary = app.flush(args.day)
    print(summary)
    return 0


def cmd_journal(args) -> int:
    from datetime import date

    app = build_app(args)
    d = date.fromisoformat(args.day) if args.day else date.today()
    print(app.journal.read_day(d) or f"（{d.isoformat()} 暂无日志）")
    return 0


def cmd_memory(args) -> int:
    app = build_app(args)
    print(app.journal.read_memory() or "（MEMORY.md 暂无内容，先跑几次执行再 flush）")
    return 0


def cmd_repl(args) -> int:
    app = build_app(args)
    print("SkillFlow REPL — 输入 help 查看命令")
    while True:
        try:
            line = input("sf> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            _repl_exec(app, line)
        except Exception as e:
            print(f"错误: {e}")


def _repl_exec(app: HarnessApp, line: str):
    parts = shlex.split(line)
    cmd, rest = parts[0], parts[1:]
    if cmd in ("help", "?"):
        print(
            "命令:\n"
            "  skills                      列出技能\n"
            "  info <skill>                技能详情\n"
            "  run <skill> [k=v ...]       执行技能\n"
            "  pipe <a|b|c> [k=v ...]      管道执行\n"
            "  chat <问题>                 自然语言入口（ReAct 元技能）\n"
            "  unload <skill>              卸载已加载的实现\n"
            "  budget [N]                  查看/设置加载预算\n"
            "  loaded                      已加载实现（L2）\n"
            "  reload                      增量重扫\n"
            "  flush                       当日日志 -> MEMORY.md\n"
            "  journal                     当日日志\n"
            "  memory                      MEMORY.md\n"
            "  sessions                    会话（Lane）状态\n"
            "  quit / exit                 退出"
        )
    elif cmd == "skills":
        for s in app.registry.list_all():
            hb = f" 心跳={s.heartbeat}" if s.heartbeat else ""
            print(f"  - {s.name} (weight={s.weight}){hb}: {s.description[:60]}")
    elif cmd == "info":
        print(json.dumps(app.info(rest[0]), ensure_ascii=False, indent=1))
    elif cmd == "run":
        if not rest:
            print("用法: run <skill> [k=v ...]")
            return
        inputs = _parse_inputs(rest[1:])
        for ev in app.run_stream("repl", {"skill": rest[0], "inputs": inputs}, on_event=print_event):
            pass
    elif cmd == "chat":
        if not rest:
            print("用法: chat <自然语言问题>")
            return
        question = " ".join(rest)
        answer = "（未产生回答）"
        for ev in app.run_stream("repl", {"skill": "agent-react", "inputs": {"question": question}}):
            if ev.kind == "stage_ok":
                out = ev.payload.get("output") or {}
                answer = out.get("answer", answer)
            print_event(ev)
        print(f"💬 {answer}")
    elif cmd == "unload":
        if not rest:
            print("用法: unload <skill>")
            return
        app.runtime.unload(rest[0])
        print(f"已卸载 {rest[0]}")
    elif cmd == "pipe":
        if not rest:
            print("用法: pipe <a|b|c> [k=v ...]")
            return
        # 管道表达式与 k=v 输入混在 rest 里：在第一个 k= 前切分
        import re as _re

        parts = _re.split(r"\s+(?=\S+=)", " ".join(rest))
        pipe_expr, kv_pairs = parts[0], parts[1:]
        inputs = _parse_inputs(kv_pairs)
        for ev in app.run_stream("repl", {"pipe": pipe_expr, "inputs": inputs}, on_event=print_event):
            pass
    elif cmd == "budget":
        if rest:
            app.runtime.budget.capacity = int(rest[0])
        print(app.runtime.budget.to_dict())
    elif cmd == "loaded":
        print("已加载实现:", app.runtime.loaded_names() or "（无 —— 所有 skill 都未被加载）")
    elif cmd == "reload":
        r = app.scan(force=True)
        print(f"变化 {len(r['changed'])}: {r['changed']}")
    elif cmd == "flush":
        print(app.flush())
    elif cmd == "journal":
        print(app.journal.read_day() or "（今日暂无日志）")
    elif cmd == "memory":
        print(app.journal.read_memory() or "（MEMORY.md 暂无内容）")
    elif cmd == "sessions":
        print(json.dumps(app.hub.list_sessions(), ensure_ascii=False, indent=1))
    elif cmd in ("quit", "exit"):
        raise SystemExit
    else:
        print(f"未知命令: {cmd}（help 查看）")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skillflow",
        description="SkillFlow — 渐进式 Skills 加载执行 Harness",
    )
    parser.add_argument("--skills-dir", default=None, help="skills 目录（默认 ../skills）")
    parser.add_argument("--budget", type=int, default=100, help="L2 加载预算（weight 上限）")

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("scan", help="增量扫描技能（只重解析变化部分）")
    p.add_argument("--force", action="store_true", help="强制全量重解析")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("info", help="技能详情（仅 L1 元数据 + L3 资源清单）")
    p.add_argument("skill")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("run", help="渐进式执行单个技能")
    p.add_argument("skill")
    p.add_argument("input", nargs="*", help="k=v 或 k=json 输入")
    p.add_argument("--session", default="cli")
    p.add_argument("--policy", default="skip", choices=["stop", "skip", "default"], help="失败策略")
    p.add_argument("--verbose", action="store_true", help="打印报告明细")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pipe", help="管道执行（契约对接，逐级产出）")
    p.add_argument("pipeline", help="例如 'fetch-source | word-count | format-report'")
    p.add_argument("input", nargs="*", help="k=v 输入")
    p.add_argument("--session", default="cli")
    p.add_argument("--policy", default="skip", choices=["stop", "skip", "default"])
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_pipe)

    p = sub.add_parser("heartbeat", help="心跳调度")
    p.add_argument("--once", action="store_true", help="立即触发一次全部心跳技能")
    p.set_defaults(func=cmd_heartbeat)

    p = sub.add_parser("watch", help="热更新监听：新增/修改技能无需重启")
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("serve", help="启动 HTTP 网关（SSE 实时事件流）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8620)
    p.add_argument("--no-watch", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("flush", help="Memory Flush：当日日志摘要写入 MEMORY.md")
    p.add_argument("--day", default=None)
    p.set_defaults(func=cmd_flush)

    p = sub.add_parser("journal", help="查看当日 Markdown 日志")
    p.add_argument("--day", default=None)
    p.set_defaults(func=cmd_journal)

    p = sub.add_parser("memory", help="查看 MEMORY.md")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("repl", help="交互式命令行").set_defaults(func=cmd_repl)

    p = sub.add_parser("chat", help="自然语言入口（ReAct 元技能调度）")
    p.add_argument("question", help="自然语言问题，如 \"统计这句话的单词数：hello world\"")
    p.add_argument("--session", default="cli")
    p.add_argument("--max-iterations", type=int, default=0, help="最大推理轮数（默认技能内 6）")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("unload", help="卸载已加载的技能实现（释放内存，下次执行重载）")
    p.add_argument("skill")
    p.set_defaults(func=cmd_unload)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
