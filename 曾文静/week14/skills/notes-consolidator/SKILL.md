---
name: notes-consolidator
description: >-
  把零散的中文学习笔记整理成去重、无损、结构化的 Markdown 笔记。
  触发："整理笔记 / 合并去重 / 笔记太乱"。
---

# 学习笔记整理器

## 流程

1. 生成精简计划（正文不进上下文）：
   ```bash
   python3 <skill_dir>/scripts/consolidate.py <data_dir>/raw_notes.md <tmp>/plan.json
   ```
2. 读 plan.json：`sections` 章节清单（仅参考）；`fuzzy_pairs` 近似重复对，
   逐对决策保留 `a` 或 `b`（取信息更全、表述更清晰的一侧）。完全重复已被脚本删除，无需处理。
3. 写决策 `decisions.json`：
   ```json
   {"fuzzy_choices": [{"pair": 0, "keep": "b"}], "section_merges": {"<章节>": "<目标章节>"}}
   ```
4. 脚本自动组装（正文从原文确定性拷贝）：
   ```bash
   python3 <skill_dir>/scripts/consolidate.py <data_dir>/raw_notes.md <tmp>/plan.json --assemble <tmp>/decisions.json -o consolidated.md
   ```
5. 无损校验，`✅ 无损` 才可交付：
   ```bash
   python3 <skill_dir>/scripts/consolidate.py <data_dir>/raw_notes.md <tmp>/plan.json --verify consolidated.md
   ```

## 原则

- 每对 fuzzy 必须给 `keep`；拿不准时保留较长一侧。
- 不编造原文没有的信息。
- 输出必须通过 `--verify`（唯一内容 0 丢失、标题全覆盖）。
