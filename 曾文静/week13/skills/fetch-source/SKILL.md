---
name: fetch-source
version: 1.0.0
description: >-
  读取一个本地资源文件并"取回"其文本内容，模拟渐进式执行的第一段：数据入口。
  声明 weight=5 演示 L2 加载预算 —— 预算不足时该技能会被推迟（deferred）。
weight: 5
consumes:
  file:
    type: str
    required: false
    default: sample.txt
    desc: 要读取的资源文件名（resources/ 目录下）
provides:
  text: 资源文件全文
  source: 来源说明（文件名 + 字节数）
deps: []
heartbeat: null
tags: [demo, l3-resource]
---

# 数据入口技能（L3 资源按需读取）

`ctx.resource(file)` 是引擎注入的 L3 资源接口：首次读取才真正把文件读进内存，
读取动作会同步产生一条 `load(L3)` 事件，让外部观察到"资源被按需加载"的过程。

执行过程中 yield `Progress`，演示渐进式执行的过程可见性。
