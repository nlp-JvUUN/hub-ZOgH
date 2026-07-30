---
name: slow-progress
version: 1.0.0
description: >-
  慢速任务模拟：逐步 yield 进度，演示渐进式执行的过程可见性
  （CLI 逐条打印、HTTP-SSE 实时推送）。
weight: 1
consumes:
  steps:
    type: int
    required: false
    default: 5
    desc: 分几步完成
  label:
    type: str
    required: false
    default: 任务
    desc: 任务名称
provides:
  summary: 完成摘要
deps: []
heartbeat: null
tags: [demo, streaming]
---

# 进度流演示技能

生成器写法：每完成一步 yield 一个 `Progress(done, total, message)`，
引擎把它转成 progress 事件转发给所有观察者 —— 结果未出，过程可见。
