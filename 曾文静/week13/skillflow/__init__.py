"""
SkillFlow — 渐进式 Skills 加载执行 Harness
===========================================

作业主题：写一套可以实现「渐进式加载执行 skills」的 harness。

「渐进式」在本项目里被拆成三条可验证的主轴，而不是一句口号：

  1. 加载渐进（Load incrementally）
     L0 目录名 -> L1 元数据（SKILL.md frontmatter）-> L2 实现（skill.py）-> L3 资源（resources/）
     只有被真正执行到的 skill 才会走到 L2/L3；manifest 缓存使重复扫描只解析
     「变化的部分」，支持运行时热更新（新增/修改 skill 无需重启）。

  2. 执行渐进（Execute progressively）
     skill 实现是生成器：执行过程中不断 yield 进度与中间结果，harness 流式转发；
     管道（pipeline）按 provides/consumes 契约把上一阶段的输出接进下一阶段的输入，
     逐级产出，失败按策略（stop/skip/default）降级，部分结果不丢。

  3. 会话渐进（Deliver progressively）
     借鉴课件 Fat Gateway 的 Lane 队列：每个会话独占 FIFO 通道，
     isRunning/hasError/retryCount 三标志保护，消息严格串行；
     另支持 HEARTBEAT 心跳技能主动自驱，以及 Markdown 记忆日志 + Memory Flush。

设计约束：
  - 零第三方依赖，纯标准库
  - 同步生成器流，不用 asyncio
  - 契约式数据对接（consumes/provides），而非「skill-name 改名即参数」的命名约定
  - 执行记录落 Markdown 日志（人机双读），而非 SQLite
"""

from .model import (
    SkillSpec,
    Progress,
    Event,
    StageResult,
    ExecutionReport,
    FrontmatterError,
    BudgetExceeded,
)
from .discovery import Registry, Manifest
from .loader import SkillRuntime, LoadBudget
from .engine import PipelineEngine
from .session import SessionHub, InternalMessage
from .journal import Journal
from .scheduler import HeartbeatScheduler

__version__ = "1.0.0"
__all__ = [
    "SkillSpec",
    "Progress",
    "Event",
    "StageResult",
    "ExecutionReport",
    "FrontmatterError",
    "BudgetExceeded",
    "Registry",
    "Manifest",
    "SkillRuntime",
    "LoadBudget",
    "PipelineEngine",
    "SessionHub",
    "InternalMessage",
    "Journal",
    "HeartbeatScheduler",
]
