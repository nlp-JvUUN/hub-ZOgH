#!/usr/bin/env python3
"""consolidate.py — notes-consolidator 最终版脚本（v3，自进化后）。

相对 v2 的进化（见 EVOLUTION.md）：
  1. 计划改为【精简版】：plan.json 只包含 LLM 需要判断的内容
     （章节清单、完全重复、近似重复对），不再携带全部正文 → LLM 输入 token 大降；
  2. 新增【模糊去重】：规范化后 difflib 相似度检测"近似重复"（v2 完全漏检）；
  3. 新增【脚本自动组装】：LLM 只输出合并决策 decisions.json，
     最终文档由脚本从原文确定性组装 → 输出 token 大降、格式稳定；
  4. --verify 无损校验：唯一内容 0 丢失 + 章节标题全覆盖。

用法：
  python3 consolidate.py <raw_notes.md> <plan.json>                       # 生成精简计划
  python3 consolidate.py <raw_notes.md> <plan.json> --assemble <decisions.json> -o out.md
  python3 consolidate.py <raw_notes.md> <plan.json> --verify out.md       # 无损校验
"""
import difflib
import json
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FUZZY_THRESHOLD = 0.72  # 归一化相似度阈值，高于此值视为近似重复


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def norm_heading(h: str) -> str:
    """标题归一化：去掉（重复）类后缀与空白，合并重复标题视为同一章节。"""
    return re.sub(r"[（(](重复|重复标题)[）)]", "", h).replace(" ", "")


def split_sections(text: str):
    """按 markdown 标题把文本切成 (heading, [lines]) 列表（不含标题行）。"""
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


def token_est(text: str) -> int:
    """本地 token 估算：CJK 字符按 1.6 字符/token，其余按 4 字符/token。"""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return round(cjk / 1.6 + other / 4)


def detect_exact_dups(lines):
    """完全重复：规范化后相同的行，保留第一处，其余记为重复。"""
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


def detect_fuzzy_dups(lines, threshold=FUZZY_THRESHOLD):
    """近似重复：两两比较内容行，相似度 ≥ 阈值且非完全重复时记为合并候选。"""
    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = norm(lines[i]), norm(lines[j])
            if not a or not b or a == b:
                continue
            if len(a) < 12 or len(b) < 12:
                continue
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r >= threshold:
                pairs.append({"pair": [i, j], "similarity": round(r, 3),
                              "text_a": lines[i].strip(), "text_b": lines[j].strip()})
    return pairs


def make_plan(src: Path, out: Path) -> int:
    text = src.read_text(encoding="utf-8")
    sections = split_sections(text)
    all_lines = [ln for _, ls in sections for ln in ls]
    exact = detect_exact_dups(all_lines)
    fuzzy = detect_fuzzy_dups(all_lines)

    plan = {
        "说明": "精简计划：只含需判断的内容。完全重复已由脚本自动删除（共 %d 处），无需处理。" % len(exact),
        "sections": [[h, len(ls)] for h, ls in sections],
        "fuzzy_pairs": [{"pair": i, "similarity": p["similarity"],
                         "a": p["text_a"], "b": p["text_b"]}
                        for i, p in enumerate(fuzzy)],
        "stats": {
            "total_lines": len(all_lines),
            "exact_dup_lines": len(exact),
            "fuzzy_pairs": len(fuzzy),
            "sections": len(sections),
        },
    }
    out.write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[v3] 精简计划: {len(sections)} 章节 | 完全重复 {len(exact)} 行 | "
          f"近似重复 {len(fuzzy)} 对 | -> {out}")
    return 0


def assemble(src: Path, plan_path: Path, decisions_path: Path, out: Path) -> int:
    """按 LLM 决策从原文确定性组装最终文档。"""
    text = src.read_text(encoding="utf-8")
    sections = split_sections(text)
    all_lines = [ln for _, ls in sections for ln in ls]
    exact = detect_exact_dups(all_lines)
    fuzzy = detect_fuzzy_dups(all_lines)
    exact_keys = {norm(d["text"]) for d in exact}
    try:
        dec = json.loads(decisions_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[assemble] ❌ decisions.json 读取失败: {e}")
        return 1
    choices = {p["pair"]: (p.get("keep", "a").lower()) for p in dec.get("fuzzy_choices", [])}
    sec_merges = {norm_heading(k): v for k, v in dec.get("section_merges", {}).items()}

    # 决策校验：choice 必须是 a/b
    for pair_no, side in choices.items():
        if side not in ("a", "b"):
            print(f"[assemble] ❌ fuzzy_pair[{pair_no}] 的 keep 必须是 'a' 或 'b'")
            return 1

    # 1) 完全重复：只保留第一次出现（seen_lines 已覆盖，exact_keys 仅用于校验）
    seen_lines = set()
    keep = []  # (section_idx, line_idx, text, is_fuzzy, fuzzy_key)
    for si, (_, ls) in enumerate(sections):
        for li, ln in enumerate(ls):
            k = norm(ln)
            if not k or k in seen_lines:
                continue
            seen_lines.add(k)
            keep.append([si, li, ln, False, k])

    # 2) 近似重复：两侧都在时，把 chosen 侧文本放到首次出现位置并删除另一侧；
    #    只剩一侧时（另一侧已被完全重复等机制移除）原样保留，不再误删。
    for pair_no, pair in enumerate(fuzzy):
        pos_a = next((i for i, x in enumerate(keep)
                      if x is not None and x[4] == norm(pair["text_a"])), None)
        pos_b = next((i for i, x in enumerate(keep)
                      if x is not None and x[4] == norm(pair["text_b"])), None)
        if pos_a is None and pos_b is None:
            continue
        if pos_a is not None and pos_b is not None:
            side = choices.get(pair_no, "a")
            chosen_text = pair["text_b"] if side == "b" else pair["text_a"]
            keep[pos_a][2] = chosen_text  # 替换首次出现处文本
            keep[pos_b] = None            # 删除另一侧
        # 只剩一侧：原样保留（该对已被完全重复等机制部分处理）

    keep = [x for x in keep if x is not None]

    # 3) 按决策合并章节（section_merges: 归一化标题 → 目标章节），并合并归一化同标题章节
    merged = {}     # norm_heading -> [lines]
    canonical = {}  # norm_heading -> 首次出现的原始标题
    for si, (head, _) in enumerate(sections):
        sec_lines = [k[2] for k in keep if k[0] == si]
        if not sec_lines:
            continue
        key = norm_heading(head)
        target = norm_heading(sec_merges.get(key, key))
        if target not in merged:
            canonical[target] = head
            merged[target] = []
        merged[target].extend(sec_lines)

    # 4) 输出（用首次出现的原始标题）
    lines = ["# 整理后的笔记：RAG 学习笔记", ""]
    for key in merged:
        lines.append(f"## {canonical[key]}")
        lines.extend(merged[key])
        lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[assemble] ✅ 已组装 {len(merged)} 个章节 -> {out}")
    return 0


def verify(raw_text: str, target: Path) -> int:
    """无损校验：唯一内容 0 丢失 + 含唯一内容的章节标题全覆盖。"""
    if not target.exists():
        print(f"[verify] ❌ 输出文件不存在: {target}")
        return 1
    out_text = target.read_text(encoding="utf-8")
    out_norm = {norm(ln) for ln in out_text.splitlines() if norm(ln)}

    sections = split_sections(raw_text)
    all_lines = [ln for _, ls in sections for ln in ls]
    exact = detect_exact_dups(all_lines)
    exact_keys = {norm(d["text"]) for d in exact}
    fuzzy = [norm(x) for p in detect_fuzzy_dups(all_lines)
             for x in (p["text_a"], p["text_b"])]

    missing = []
    for ln in all_lines:
        k = norm(ln)
        if not k:
            continue
        if k in exact_keys:          # 完全重复：至少保留一处
            if k not in out_norm:
                missing.append(ln)
            continue
        if k in out_norm:
            continue
        if k in fuzzy and fuzzy[fuzzy.index(k) ^ 1] in out_norm:
            continue  # 近似对任一侧保留即视为已合并
        missing.append(ln)

    # 标题覆盖：只要求"含非重复内容"的章节标题（归一化后）出现
    required = {norm_heading(h) for h, ls in sections
                if any(norm(x) not in exact_keys and norm(x) not in fuzzy for x in ls)}
    heads_out = {norm_heading(m.group(2).strip())
                 for m in (HEADING_RE.match(l) for l in out_text.splitlines()) if m}
    miss_heads = required - heads_out

    ok = not missing and not miss_heads
    print(f"[verify] 唯一内容丢失 {len(missing)} 处 | 标题缺失 {len(miss_heads)} 个 | "
          f"{'✅ 无损' if ok else '❌ 有损'}")
    for ln in missing[:5]:
        print(f"   - 丢失: {ln[:60]}")
    for h in miss_heads:
        print(f"   - 标题缺失: {h}")
    return 0 if ok else 1


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 1
    src, plan_path = Path(argv[0]), Path(argv[1])

    if "--assemble" in argv:
        i = argv.index("--assemble")
        decisions = Path(argv[i + 1])
        out = Path(argv[argv.index("-o") + 1]) if "-o" in argv else Path("consolidated.md")
        return assemble(src, plan_path, decisions, out)
    if "--verify" in argv:
        return verify(src.read_text(encoding="utf-8"), Path(argv[argv.index("--verify") + 1]))
    return make_plan(src, plan_path)


if __name__ == "__main__":
    sys.exit(main())
