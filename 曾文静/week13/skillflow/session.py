"""
会话层 — 借鉴课件「Fat Gateway」的 Lane 队列：每个会话独占 FIFO 通道。

课件要点（slide 11）：
  - 没有 Lane 时，同一会话的两条消息可能被两个线程同时处理，
    写回互相覆盖 → 逻辑错乱；Lane 队列让同一会话的消息严格串行。
  - 三个状态保护标志：isRunning（处理中 → 新消息自动排队）、
    hasError（上轮出错 → 暂停等确认）、retryCount（失败超限 → 整条 Lane 暂停）。

本模块把这三个概念落成代码：
  - SessionHub.submit() 统一消息入口（InternalMessage 归一化，类比 Channel Adapter）
  - 每个 Session 一个 Lane（deque）+ 一个工作线程，严格 FIFO 串行
  - lane 三标志：is_running / has_error / retry_count；失败达上限自动 pause，
    可手动 resume（等"用户确认"）
  - 事件全局可见：会话内产生的事件全部进入 hub 的事件总线，
    HTTP 网关的 SSE 就是从这个总线上按 session 订阅

这里把「harness 作为服务」的形态做出来 —— 渐进式执行对上层表现为
「投递消息 → 流式收事件」。
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .model import Event


@dataclass
class InternalMessage:
    """
    归一化后的内部消息（Channel Adapter 的输出）。
    字段对齐课件 InternalMessage 的骨架：sessionId / content / metadata。
    """

    session_id: str
    content: Dict[str, Any]  # {"skill": ...} 或 {"pipe": [...]} + {"inputs": {...}, "config": {...}}
    msg_id: str = ""
    channel: str = "api"
    ts: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = uuid.uuid4().hex[:12]
        if not self.ts:
            self.ts = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "session_id": self.session_id,
            "channel": self.channel,
            "content": self.content,
            "ts": round(self.ts, 3),
        }


class Session:
    """一个会话 = 一条 Lane。"""

    def __init__(self, session_id: str, max_retries: int = 3):
        self.session_id = session_id
        self.lane: "queue.Queue[InternalMessage]" = queue.Queue()
        self.max_retries = max_retries

        # Lane 三标志（课件 slide 11）
        self.is_running = False
        self.has_error = False
        self.retry_count = 0
        self.paused = False

        self.created = time.time()
        self.processed: List[InternalMessage] = []  # 已处理消息（可重放）
        self.processed_ids: set = set()
        self.current: Optional[InternalMessage] = None  # 正在处理/待重试的消息
        self._wake = threading.Event()

    def enqueue(self, msg: InternalMessage) -> int:
        """入队，返回队列深度。"""
        self.lane.put(msg)
        self._wake.set()
        return self.lane.qsize()

    def resume(self) -> bool:
        """用户确认后恢复 Lane（hasError/paused -> False，重试当前消息）。"""
        if not (self.has_error or self.paused):
            return False
        self.has_error = False
        self.paused = False
        self.retry_count = 0
        self._wake.set()
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "is_running": self.is_running,
            "has_error": self.has_error,
            "retry_count": self.retry_count,
            "paused": self.paused,
            "queue_depth": self.lane.qsize(),
            "processed": len(self.processed),
            "created": round(self.created, 3),
        }


class SessionHub:
    """
    会话中枢：负责 Lane 的生命周期、串行消费、事件总线。

    用法：
        hub = SessionHub(processor=my_processor)   # processor(msg) -> [Event...]
        hub.submit("s1", {"skill": "word-count", "inputs": {...}})
        hub.events("s1", after=0)                  # 轮询增量事件
    """

    def __init__(
        self,
        processor: Callable[[InternalMessage, Callable[[Event], None]], List[Event]],
        max_retries: int = 3,
    ):
        self.processor = processor
        self.max_retries = max_retries
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        # 事件总线：session -> [Event]，全部会话共享一个序号，便于增量拉取
        self._events: List[Event] = []
        self._events_lock = threading.Lock()

    # ── 会话管理 ─────────────────────────────────────────────

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(session_id, max_retries=self.max_retries)
                self._sessions[session_id] = s
                self._start_worker(s)
            return s

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.snapshot() for s in self._sessions.values()]

    def _start_worker(self, session: Session):
        t = threading.Thread(target=self._drain, args=(session,), daemon=True, name=f"lane-{session.session_id}")
        t.start()

    # ── 消息入口（Channel Adapter 之后） ─────────────────────

    def submit(self, session_id: str, content: Dict[str, Any], channel: str = "api") -> InternalMessage:
        msg = InternalMessage(session_id=session_id, content=content, channel=channel)
        session = self.get_session(session_id)
        depth = session.enqueue(msg)
        msg.metadata["depth"] = depth
        if depth > 1:
            self.publish(
                Event("discover", session_id, "__lane__", {"message": f"消息 {msg.msg_id} 已入队等待（队列深度 {depth}）"})
            )
        return msg

    # ── Lane 消费循环：严格串行 ──────────────────────────────

    def _drain(self, session: Session):
        while True:
            session._wake.wait()
            session._wake.clear()
            if session.paused or session.has_error:
                continue  # 暂停/出错中：等 resume 确认
            if session.current is None:
                try:
                    session.current = session.lane.get_nowait()
                except queue.Empty:
                    continue

            msg = session.current
            session.is_running = True
            extra: List[Event] = []
            try:
                extra = self.processor(msg, self.publish) or []
                session.current = None  # 只有成功才出队
                session.retry_count = 0
                session.processed.append(msg)
                session.processed_ids.add(msg.msg_id)
            except Exception as e:
                # 自动重试同一消息；连续失败超限 -> hasError + 暂停 Lane（课件三标志）
                session.retry_count += 1
                extra = [
                    Event(
                        "stage_fail",
                        session.session_id,
                        "__lane__",
                        {"error": f"Lane 处理失败（第 {session.retry_count}/{session.max_retries} 次）: {e}"},
                    )
                ]
                if session.retry_count >= session.max_retries:
                    session.has_error = True
                    session.paused = True
                    extra.append(
                        Event(
                            "report",
                            session.session_id,
                            "__lane__",
                            {"status": "failed", "message": f"重试 {session.max_retries} 次仍失败，Lane 已暂停，请 resume 确认后继续"},
                        )
                    )
            finally:
                session.is_running = False

            for ev in extra:
                self.publish(ev)
            session._wake.set()  # 处理完唤醒，继续取队列中的下一条

    # ── 事件总线 ─────────────────────────────────────────────

    def publish(self, event: Event):
        with self._events_lock:
            self._events.append(event)

    def events(self, session_id: Optional[str] = None, after: int = 0) -> List[Dict[str, Any]]:
        """增量拉取事件（HTTP 网关的 SSE 轮询就用它）。"""
        with self._events_lock:
            batch = self._events[after:]
            if session_id is None:
                return [e.to_dict() for e in batch]
            return [e.to_dict() for e in batch if e.session == session_id]

    def event_count(self) -> int:
        with self._events_lock:
            return len(self._events)

    def wait_for(self, msg: InternalMessage, timeout: float = 15.0) -> bool:
        """
        阻塞直到某条消息被 Lane 处理完毕（成功）。
        CLI 的 heartbeat --once 等场景用：避免进程在 daemon 线程处理前退出。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            session = self._sessions.get(msg.session_id)
            if session is not None and msg.msg_id in session.processed_ids:
                return True
            time.sleep(0.05)
        return False
