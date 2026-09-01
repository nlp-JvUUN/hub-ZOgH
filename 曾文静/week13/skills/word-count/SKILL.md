---
name: word-count
version: 1.0.0
description: 统计一段文本的单词数、字符数与高频词（纯函数式技能，最轻量）。
weight: 1
consumes:
  text:
    type: str
    required: true
    desc: 要统计的文本内容
  top_n:
    type: int
    required: false
    default: 5
    desc: 输出多少个高频词
provides:
  count: 单词总数
  chars: 字符总数
  top_words: 高频词列表
deps: []
heartbeat: null
tags: [demo, text]
---

# 词频统计技能

演示最轻量的技能写法：普通函数直接返回 dict 即可，无需生成器。
管道示例中它消费 fetch-source 提供的 `text`（契约对接）。
