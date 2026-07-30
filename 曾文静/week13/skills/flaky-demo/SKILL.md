---
name: flaky-demo
version: 1.0.0
description: >-
  按需失败的演示技能：should_fail=true 时抛异常，用于验证失败策略
  （stop 整体停止 / skip 级联跳过 / default 默认值兜底）。
weight: 1
consumes:
  should_fail:
    type: bool
    required: false
    default: false
    desc: 是否故意失败
  fallback:
    type: str
    required: false
    default: 默认值兜底报告
    desc: default 策略下使用的兜底输出
provides:
  status: 状态
  report: 报告文本
deps: []
heartbeat: null
tags: [demo, failure]
---

# 失败注入演示技能

配合 `--policy` 参数观察三种失败策略的行为差异：
- stop：失败即中止，后续 stage 不执行
- skip：跳过本 stage，下游若依赖它则级联跳过
- default：用 SKILL.md 声明的默认值继续执行
