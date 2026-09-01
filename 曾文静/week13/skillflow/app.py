"""
HarnessApp — 组装层：把发现/加载/执行/会话/日志/心跳接成一台完整的 harness。

CLI、REPL、HTTP 网关都复用这一个类 —— 三种入口看到的是同一套状态
（同一个 manifest 缓存、同一个加载预算、同一个事件总线、同一本日志）。

process 消息协议（HTTP/CLI/心跳共用）：
    {"skill": "word-count", "inputs": {...}, "config": {...}}
    {"pipe": "fetch-source | word-count | format-report", "inputs": {...}, "config": {...}}
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .discovery import Manifest, Registry
from .engine import PipelineEngine, parse_pipeline
from .journal import Journal
from .loader import LoadBudget, SkillRuntime
from .model import BudgetExceeded, Event, ExecutionReport
from .scheduler import HEARTBEAT_SESSION, HeartbeatScheduler
from .session import InternalMessage, SessionHub


class HarnessApp:
    def __init__(
        self,
        skills_dir: Path,
        state_dir: Optional[Path] = None,
        journal_dir: Optional[Path] = None,
        budget: int = 100,
        auto_scan: bool = True,
    ):
        self.skills_dir = Path(skills_dir)
        self.state_dir = Path(state_dir) if state_dir else self.skills_dir.parent / "state"
        self.journal = Journal(Path(journal_dir) if journal_dir else self.skills_dir.parent / "journal")

        self.manifest = Manifest(self.skills_dir, self.state_dir)
        self.registry = Registry(self.manifest)
        self.runtime = SkillRuntime(LoadBudget(capacity=budget))
        self.engine = PipelineEngine(self.registry, self.runtime)

        self.hub = SessionHub(self._process_message)
        self.scheduler = HeartbeatScheduler(self.registry, self._submit_heartbeat)

        self._watcher: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()

        if auto_scan:
            self.scan(force=False)

    # ── 发现 ─────────────────────────────────────────────────

    def scan(self, force: bool = False) -> Dict[str, Any]:
        specs, changed = self.manifest.scan(force=force)
        self.scheduler.refresh_schedule()
        deps_errors = self.registry.check_deps()
        return {
            "skills": [s.to_dict() for s in specs],
            "changed": changed,
            "deps_errors": deps_errors,
            "budget": self.runtime.budget.to_dict(),
            "loaded_impls": self.runtime.loaded_names(),
        }

    def info(self, name: str) -> Dict[str, Any]:
        spec = self.registry.require(name)
        data = spec.to_dict()
        data["resources"] = self.runtime.list_resources(spec)  # L3 清单（不读内容）
        data["impl_loaded"] = self.runtime.is_loaded(name)  # L2 是否已加载
        return data

    # ── 执行 ─────────────────────────────────────────────────

    def run_stream(
        self,
        session: str,
        content: Dict[str, Any],
        on_event: Optional[Callable[[Event], None]] = None,
    ) -> List[Event]:
        """同步跑一次（CLI/REPL 用）：边执行边回调事件，返回全部事件。"""
        events: List[Event] = []
        for ev in self._iter_pipeline_events(session, content):
            events.append(ev)
            self.journal.log_event(ev)
            if on_event:
                on_event(ev)
        return events

    def _system_services(self) -> Dict[str, Any]:
        """harness 注入给技能的系统服务（通过 ctx.system 取用）。"""
        return {
            "journal_dir": str(self.journal.base_dir),
            "skills_dir": str(self.skills_dir),
            "list_skills": self._list_skills_service,
            "execute_skill": self._execute_skill_service,
        }

    def _iter_pipeline_events(self, session: str, content: Dict[str, Any]):
        """把消息内容翻译成 engine.run 的调用。"""
        names, inputs, config = self._parse_content(content)
        system = config.setdefault("system", {})
        system.update({k: v for k, v in self._system_services().items() if k not in system})
        yield from self.engine.run(session, names, inputs, config)

    # ── 注入给技能的系统服务（L1 视图 + 受控执行器） ─────────

    def _list_skills_service(self) -> List[Dict[str, Any]]:
        """供 agent-react 等元技能使用：只读 L1 元数据，不加载任何实现。"""
        out = []
        for spec in self.registry.list_all():
            d = spec.to_dict()
            d.pop("notes", None)
            out.append(d)
        return out

    def _execute_skill_service(self, name: str, params: Dict[str, Any]) -> Any:
        """供元技能调用：执行另一个技能并返回其输出；任何失败都抛异常。"""
        if name == "agent-react":
            raise ValueError("禁止递归调用 agent-react")
        events = list(
            self.engine.run("__agent__", [name], dict(params or {}), {"system": self._system_services()})
        )
        report = events[-1].payload
        for st in reversed(report.get("stages", [])):
            if st["status"] == "ok":
                return st["output"]
        raise RuntimeError(f"技能 {name} 未成功执行: {report.get('message', '')}")

    @staticmethod
    def _parse_content(content: Dict[str, Any]) -> tuple:
        if "skill" in content:
            names = [str(content["skill"])]
        elif "pipe" in content:
            names = parse_pipeline(str(content["pipe"]))
        elif "pipeline" in content:
            names = [str(x) for x in content["pipeline"]]
        else:
            raise ValueError("消息内容需要 skill / pipe / pipeline 之一")
        inputs = content.get("inputs") or {}
        config = content.get("config") or {}
        return names, inputs, config

    # ── SessionHub 处理器（Lane 内执行 + 实时发布 + 落日志） ─

    def _process_message(self, msg: InternalMessage, publish: Callable[[Event], None]) -> List[Event]:
        """
        会话 Lane 的处理器：实时发布事件（SSE 能看到过程），并写日志。
        """
        events: List[Event] = []
        try:
            for ev in self._iter_pipeline_events(msg.session_id, msg.content):
                events.append(ev)
                self.journal.log_event(ev)
                publish(ev)
        except Exception as e:
            ev = Event("stage_fail", msg.session_id, "__lane__", {"error": f"{type(e).__name__}: {e}"})
            events.append(ev)
            self.journal.log_event(ev)
            publish(ev)
        return []  # 已实时发布，无需 _drain 再发

    def _submit_heartbeat(self, skill_name: str):
        return self.hub.submit(HEARTBEAT_SESSION, {"skill": skill_name}, channel="heartbeat")

    # ── 记忆 ─────────────────────────────────────────────────

    def flush(self, day: Optional[str] = None) -> str:
        from datetime import date

        d = date.fromisoformat(day) if day else date.today()
        summary = self.journal.flush(d)
        self.hub.publish(Event("heartbeat", HEARTBEAT_SESSION, "daily-report", {"flushed": d.isoformat()}))
        return summary

    # ── 热更新 watch ─────────────────────────────────────────

    def start_watch(self, interval: float = 1.0):
        """后台线程：轮询增量扫描 + 心跳调度；有新技能/修改即打印。"""
        if self._watcher is not None:
            return

        def loop():
            for changed in self.manifest.watch(interval=interval, stop_event=self._watch_stop):
                for name in changed:
                    spec = self.registry.get(name)
                    detail = f"（删除）" if spec is None else f"weight={spec.weight} heartbeat={spec.heartbeat}"
                    self.hub.publish(
                        Event("discover", "__watch__", name, {"message": f"热更新: {name} {detail}"})
                    )
                self.scheduler.refresh_schedule()

        self._watcher = threading.Thread(target=loop, daemon=True, name="watch")
        self._watcher.start()

    def stop_watch(self):
        self._watch_stop.set()
        if self._watcher:
            self._watcher.join(timeout=3)
            self._watcher = None

    # ── 汇总视图 ─────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "skills_count": len(self.registry.list_all()),
            "budget": self.runtime.budget.to_dict(),
            "loaded_impls": self.runtime.loaded_names(),
            "sessions": self.hub.list_sessions(),
            "events_total": self.hub.event_count(),
            "journal_days": self.journal.list_days(),
        }
