---
name: format-report
version: 1.0.0
description: 把统计结果排版成一段可读报告文本（管道终端技能）。
weight: 1
consumes:
  count:
    type: int
    required: true
    desc: 单词总数（管道里由 word-count 提供）
  chars:
    type: int
    required: false
    default: 0
    desc: 字符总数
  top_words:
    type: list
    required: false
    default: []
    desc: 高频词列表
provides:
  report: 排版后的报告文本
deps: []
heartbeat: null
tags: [demo, text]
---

# 报告排版技能

管道末段：把上游（word-count）按契约提供的 count/chars/top_words
组装成人类可读的报告。这里故意把 `text` 排除在必填之外 ——
上游被跳过/推迟时，本技能仍能用默认值兜底（default 策略的演示点）。
