"""
数据模型层 — SkillFlow 里所有组件共享的类型定义。

设计说明：
  - SkillSpec 是「一份 skill」的完整描述：元数据来自 SKILL.md frontmatter（L1），
    实现与资源路径指向磁盘（L2/L3 懒加载的入口）。
  - Event / StageResult / ExecutionReport 是「渐进式执行」的对外协议：
    harness 不吞结果，而是把发现、加载、进度、成败一条条流出来，
    调用方（CLI / REPL / HTTP-SSE）按自己的节奏消费。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class FrontmatterError(ValueError):
    """SKILL.md frontmatter 解析失败。"""


class BudgetExceeded(ValueError):
    """加载预算不足，skill 被推迟（deferred），不进入执行阶段。"""


# ─────────────────────────────────────────────────────────────────────
# Skill 描述
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SkillSpec:
    """
    一个 skill 的完整规格（= 元数据 + 磁盘位置）。
      - 用 consumes/provides 声明「数据契约」，执行引擎按契约对接管道，
      - weight 声明加载代价（L2/L3 需要多少"预算"），配合 LoadBudget 实现
        「预算不够就先不加载」的渐进策略。
      - heartbeat 声明心跳周期，由 HeartbeatScheduler 主动调度（课件 HEARTBEAT.md 概念）。
    """

    name: str
    version: str
    description: str
    weight: int = 1
    consumes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 输入契约 {key: {type, required, default, desc}}
    provides: Dict[str, str] = field(default_factory=dict)  # 输出契约 {key: desc}
    deps: List[str] = field(default_factory=list)  # 需要先执行的 skill
    heartbeat: Optional[str] = None  # "30s" / "5m" / "1h" / "daily 23:59"
    tags: List[str] = field(default_factory=list)
    notes: str = ""  # SKILL.md 正文前 N 字符，用于 info 展示
    dir: Optional[Path] = None
    impl_file: Optional[Path] = None  # 约定 skill.py
    resources_dir: Optional[Path] = None  # 约定 resources/

    # 供 manifest 增量扫描用的指纹
    md5: str = ""
    mtime_ns: int = 0

    # ── 契约工具 ──────────────────────────────────────────────

    def required_inputs(self) -> List[str]:
        return [k for k, v in self.consumes.items() if v.get("required")]

    def optional_inputs(self) -> List[str]:
        return [k for k, v in self.consumes.items() if not v.get("required")]

    def default_for(self, key: str) -> Any:
        v = self.consumes.get(key, {})
        return v.get("default")

    def is_heavy(self) -> bool:
        return self.weight >= 5

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dir"] = str(self.dir) if self.dir else None
        d["impl_file"] = str(self.impl_file) if self.impl_file else None
        d["resources_dir"] = str(self.resources_dir) if self.resources_dir else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], skill_dir: Optional[Path] = None) -> "SkillSpec":
        """从 dict（frontmatter 解析结果）构造规格。"""
        if not data.get("name"):
            raise FrontmatterError("frontmatter 缺少 name")
        spec = cls(
            name=str(data["name"]).strip(),
            version=str(data.get("version", "1.0")),
            description=str(data.get("description", "")).strip(),
            weight=int(data.get("weight", 1)),
            consumes=dict(data.get("consumes", {}) or {}),
            provides=dict(data.get("provides", {}) or {}),
            deps=[str(x).strip() for x in (data.get("deps", []) or [])],
            heartbeat=data.get("heartbeat"),
            tags=[str(x).strip() for x in (data.get("tags", []) or [])],
        )
        if skill_dir is not None:
            spec.dir = Path(skill_dir)
            spec.impl_file = Path(skill_dir) / "skill.py"
            spec.resources_dir = Path(skill_dir) / "resources"
        return spec


# ─────────────────────────────────────────────────────────────────────
# 渐进式执行的事件协议
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Progress:
    """skill 生成器执行过程中 yield 的进度（渐进式输出的最小单元）。"""

    done: int = 0
    total: int = 1
    message: str = ""

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return max(0, min(100, int(self.done * 100 / self.total)))

    def to_dict(self) -> Dict[str, Any]:
        return {"done": self.done, "total": self.total, "percent": self.percent, "message": self.message}


@dataclass
class StageResult:
    """单个 skill 阶段的执行结果（成功、失败或被跳过）。"""

    skill: str
    status: str  # ok | failed | skipped | deferred
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0


class Event:
    """
    一次执行生命周期中的事件（发现/加载/进度/结果/失败/跳过/心跳）。

    
    字段扁平化为 session + kind + payload，方便 SSE 直接序列化。
    """

    KINDS = {"discover", "load", "progress", "stage_ok", "stage_fail", "stage_skip", "stage_defer", "report", "heartbeat"}

    def __init__(
        self,
        kind: str,
        session: str,
        skill: str = "",
        payload: Any = None,
        ts: Optional[float] = None,
    ):
        assert kind in self.KINDS, f"unknown event kind: {kind}"
        self.kind = kind
        self.session = session
        self.skill = skill
        self.payload = payload
        self.ts = ts if ts is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload
        if hasattr(payload, "to_dict"):  # 如 Progress 对象
            payload = payload.to_dict()
        return {
            "ts": round(self.ts, 3),
            "session": self.session,
            "kind": self.kind,
            "skill": self.skill,
            "payload": payload,
        }

    def __repr__(self) -> str:  # 让 CLI / 日志直接可读
        if self.kind == "progress":
            p = self.payload
            return f"[{self.skill}] {p.percent}% ({p.done}/{p.total}) {p.message}".rstrip()
        if self.kind == "stage_ok":
            return f"[{self.skill}] ok {self.payload.get('duration_ms', 0):.0f}ms"
        if self.kind == "stage_fail":
            return f"[{self.skill}] FAILED: {self.payload.get('error', '')}"
        if self.kind == "stage_skip":
            return f"[{self.skill}] skipped: {self.payload.get('reason', '')}"
        if self.kind == "stage_defer":
            return f"[{self.skill}] deferred: {self.payload.get('reason', '')}"
        if self.kind == "report":
            return f"[report] {self.payload.get('status')} in {self.payload.get('duration_ms', 0):.0f}ms"
        if self.kind == "load":
            return f"[{self.skill}] loaded {self.payload.get('stage', '?')}"
        if self.kind == "heartbeat":
            return f"[heartbeat] {self.skill}"
        return f"[{self.kind}] {self.skill}"


@dataclass
class ExecutionReport:
    """一次 run/pipe 的汇总报告（执行渐进的结果容器）。"""

    session: str
    status: str  # ok | failed | partial | deferred
    stages: List[StageResult]
    events: List[Event]
    duration_ms: float = 0.0
    message: str = ""

    def outputs(self) -> Dict[str, Any]:
        """最后一个成功阶段的输出（管道终端产物）。"""
        for st in reversed(self.stages):
            if st.status == "ok":
                return st.output if isinstance(st.output, dict) else {"value": st.output}
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session,
            "status": self.status,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 3),
            "stages": [
                {
                    "skill": s.skill,
                    "status": s.status,
                    "output": s.output,
                    "error": s.error,
                    "duration_ms": round(s.duration_ms, 3),
                }
                for s in self.stages
            ],
        }
