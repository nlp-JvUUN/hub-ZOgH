---
name: daily-report
version: 1.0.0
description: >-
  心跳技能：把当日执行日志提炼（Memory Flush）写入 journal/MEMORY.md，
  让 harness 从"被动响应"变成"主动行动"（课件 HEARTBEAT.md 概念）。
weight: 1
consumes: {}
provides:
  report: 刷新摘要文本
  day: 刷新日期
deps: []
heartbeat: 30s
tags: [demo, heartbeat, memory]
---

# 心跳技能（Memory Flush）

在 SKILL.md 声明 `heartbeat: 30s` 后，HeartbeatScheduler 会定期把本技能
投递到 `__heartbeat__` 会话的 Lane 执行 —— 与用户消息走完全相同的执行路径。

它读取 harness 通过 ctx.system 注入的日志目录，把当天 JSONL 事件
聚合摘要写入 MEMORY.md（规则式 Memory Flush，零依赖、可复现）。
