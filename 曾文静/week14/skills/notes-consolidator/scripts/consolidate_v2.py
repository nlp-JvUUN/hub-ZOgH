#!/usr/bin/env python3
"""consolidate_v2.py — notes-consolidator 第一轮优化脚本（v2）。

职责（把"机械性"工作从 LLM 挪到本地脚本）：
  1. 读取原始笔记，按标题切分成章节；
  2. 检测【完全重复】的句子/段落（规范化空白后逐行比对）；
  3. 输出 plan.json：仅包含"去重后的唯一内容 + 重复报告 + 统计"，供 LLM 阅读。

已知局限（v3 修复）：只检测完全重复，检测不到"近似重复"（意思相同、用词不同）。
"""
import json
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def norm(s: str) -> str:
    """规范化：去首尾空白、压缩内部空白、统一全半角空格。"""
    return re.sub(r"\s+", " ", s.strip())


def split_sections(text: str):
    """按 markdown 标题把文本切成 (heading, [lines]) 列表。"""
    sections, cur_head, cur_lines = [], None, []
    for raw in text.splitlines():
        m = HEADING_RE.match(raw)
        if m:
            if cur_head is not None or cur_lines:
                sections.append((cur_head or "(无标题前置内容)", cur_lines))
            cur_head, cur_lines = m.group(2).strip(), []
        else:
            if raw.strip():
                cur_lines.append(raw)
    if cur_head is not None or cur_lines:
        sections.append((cur_head or "(无标题前置内容)", cur_lines))
    return sections


def detect_exact_dups(lines):
    """检测完全重复行：规范化后相同的内容只保留第一处，其余记为重复。"""
    seen, dups = {}, []
    for i, ln in enumerate(lines):
        key = norm(ln)
        if not key:
            continue
        if key in seen:
            dups.append({"first": seen[key], "dup": i, "text": ln.strip()})
        else:
            seen[key] = i
    return dups


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw_notes.md")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "plan.json")
    text = src.read_text(encoding="utf-8")

    sections = split_sections(text)
    all_lines = [ln for _, ls in sections for ln in ls]
    dups = detect_exact_dups(all_lines)

    # 去重后的唯一内容（按章节组织）
    unique_sections = []
    dup_texts = {d["text"] for d in dups}
    for head, ls in sections:
        keep = [ln for ln in ls if norm(ln) not in dup_texts or ls.index(ln) == 0]
        # 精确去重：同一章节内也只保留首处
        seen_in_sec, keep2 = set(), []
        for ln in keep:
            k = norm(ln)
            if k not in seen_in_sec:
                keep2.append(ln)
                seen_in_sec.add(k)
        unique_sections.append({"heading": head, "lines": keep2})

    plan = {
        "source": str(src),
        "说明": "这是预处理结果，不是最终笔记。请只阅读本文件即可，无需再读原文。",
        "sections": unique_sections,
        "exact_duplicates": dups,
        "stats": {
            "total_lines": len(all_lines),
            "exact_dup_lines": len(dups),
            "unique_lines": len(all_lines) - len(dups),
            "sections": len(sections),
        },
    }
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v2] 预处理完成: {len(sections)} 个章节, "
          f"检测到 {len(dups)} 个完全重复行, 结果 -> {out}")


if __name__ == "__main__":
    main()
