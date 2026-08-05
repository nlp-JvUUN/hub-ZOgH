---
name: notes-consolidator
description: >-
  把零散的中文学习笔记整理成结构清晰、去重、保留全部信息的 Markdown 笔记。
  触发词："整理笔记 / 合并去重 / 笔记太乱"。
---

# 学习笔记整理器

## 流程

1. 运行预处理脚本（脚本负责切分章节、检测完全重复）：
   ```bash
   python3 <skill_dir>/scripts/consolidate_v2.py <data_dir>/raw_notes.md <输出>/plan.json
   ```
2. 用 Read 读取 plan.json（不要读原文）。
3. 按 plan.json 的 sections 组织输出：把重复标题的章节合并到一处，
   删除 exact_duplicates 中列出的重复行。
4. 输出 `consolidated.md`：全部唯一内容 + 章节合并 + 重复行已删。
5. 待办（TODO/占位）原样保留。

## 输出要求

- 只保留 plan.sections 中列出的内容，删除 exact_duplicates 中的行；
- 同一主题多个章节合并为一个，标题取最规范的一个；
- 要点列表（`- xxx`）合并同类项。
