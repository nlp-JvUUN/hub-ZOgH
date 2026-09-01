"""
心跳调度层 — 让 harness 从「被动响应」变成「主动行动」（课件 HEARTBEAT.md）。

课件要点（slide 24）：
  - 定时器触发 -> 读 HEARTBEAT.md -> Agent 自主执行；用户不在线也持续工作；
  - 为什么比 cron 强：触发的是「指令」，执行与否、如何执行由 Agent 判断。
    这里收敛为可验证的最小闭环：skill 在 SKILL.md 里声明 heartbeat，
    调度器到点把消息投进 __heartbeat__ 会话的 Lane，与普通消息一样串行执行，
    事件照常进日志 —— 心跳技能和用户消息技能走同一条执行路径。

支持两种周期：
  - 间隔型："30s" / "5m" / "1h"（进程内到点触发）
  - 每日型："daily 23:59"（每天指定时刻触发）
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

from .discovery import Registry
from .model import Event, SkillSpec
from .session import InternalMessage

HEARTBEAT_SESSION = "__heartbeat__"

_INTERVAL_RE = re.compile(r"^(\d+)\s*(s|m|h)$")
_DAILY_RE = re.compile(r"^daily\s+(\d{1,2}):(\d{2})$")


def parse_interval(expr: str) -> float:
    """把 '30s'/'5m'/'1h' 解析成秒；'daily HH:MM' 解析成距下一次的秒数。"""
    m = _INTERVAL_RE.match(expr.strip().lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return n * {"s": 1, "m": 60, "h": 3600}[unit]
    m = _DAILY_RE.match(expr.strip().lower())
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()
    raise ValueError(f"无法解析心跳周期: {expr!r}（支持 '30s'/'5m'/'1h'/'daily 23:59'）")


class HeartbeatScheduler:
    """
    心跳调度器：轮询技能的心跳声明，到点把消息投递到 __heartbeat__ Lane。

    ticker: 到点后的投递函数，形如 submit_heartbeat(skill_name) -> None
            （由调用方接 SessionHub，让心跳走与用户消息相同的串行执行路径）
    """

    def __init__(self, registry: Registry, submit_heartbeat: Callable[[str], None], interval: float = 1.0):
        self.registry = registry
        self.submit_heartbeat = submit_heartbeat
        self.poll_interval = interval
        self._stop = threading.Event()
        self._next_due: Dict[str, float] = {}  # skill -> 下次到期时间戳
        self._lock = threading.Lock()

    def heartbeat_skills(self) -> List[SkillSpec]:
        return [s for s in self.registry.list_all() if s.heartbeat]

    def refresh_schedule(self):
        """重算所有心跳技能的下次到期时间（新增技能后调用）。"""
        with self._lock:
            now = time.time()
            self._next_due = {
                s.name: now + parse_interval(s.heartbeat) for s in self.heartbeat_skills()
            }

    def run_forever(self):
        """后台线程主循环：到点投递，永不退出（供 serve/watch 使用）。"""
        self.refresh_schedule()
        while not self._stop.is_set():
            self.tick_once()
            self._stop.wait(self.poll_interval)

    def tick_once(self) -> List[str]:
        """跑一轮：投递所有到点的心跳技能，返回本次投递名单。"""
        with self._lock:
            if not self._next_due:
                self.refresh_schedule()
            now = time.time()
            due = [name for name, t in self._next_due.items() if t <= now]
            for name in due:
                spec = self.registry.get(name)
                self._next_due[name] = now + parse_interval(spec.heartbeat) if spec else now + 3600
        for name in due:
            self.submit_heartbeat(name)
        return due

    def run_due_now(self) -> List[InternalMessage]:
        """立即触发所有心跳技能一次（演示/测试用），返回投递的消息列表。"""
        msgs = []
        for s in self.heartbeat_skills():
            msgs.append(self.submit_heartbeat(s.name))
        return msgs

    def stop(self):
        self._stop.set()

    def next_dues(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._next_due)
