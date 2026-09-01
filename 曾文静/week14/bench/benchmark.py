#!/usr/bin/env python3
"""benchmark.py — notes-consolidator 优化前后基准测试。

对 v1 / v2 / v3 三版 skill 在同一份语料上测量：
  * 固定开销：SKILL.md 字符数与 token 数（每次调用都会加载）
  * 单次任务：LLM 输入（v1=全文 / v2、v3=预处理 plan）与输出（模型产出）的 token 数
  * 执行效率：本地脚本实测耗时(ms)；LLM 推理时长按 token 线性估算（标注为估算）
  * 质量：无损校验（唯一内容丢失数）、近似重复残留数（越低越好）、标题覆盖

用法：
  PYTHONPATH=/path/to/tiktoken python3 benchmark.py
输出：bench/results.json（原始测量数据）
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    TOKENIZER = "tiktoken/cl100k_base"
except ImportError:
    ENC = None
    TOKENIZER = "fallback(CJK/1.6 + other/4)"

WEEK14 = Path(__file__).resolve().parent.parent
SKILL_DIR = WEEK14 / "skills" / "notes-consolidator"
DATA = SKILL_DIR / "data" / "raw_notes.md"
SCRIPTS = SKILL_DIR / "scripts"
OUTPUTS = WEEK14 / "outputs"
TMP = WEEK14 / "bench" / "tmp"
TMP.mkdir(exist_ok=True)

LLM_MS_PER_1K_TOKENS = 1200  # 典型中档模型生成/处理速率（估算用），1.2s/1k token
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def count_tokens(text: str) -> int:
    if ENC:
        return len(ENC.encode(text))
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return round(cjk / 1.6 + (len(text) - cjk) / 4)


def run_script(cmd, cwd) -> tuple[float, str]:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    return ms, (r.stdout + r.stderr).strip()


def load_fuzzy_pairs():
    """近似重复对（与具体版本无关，由语料决定）：flat list，第 2i/2i+1 项为一对。"""
    p = json.loads((TMP / "plan_v3.json").read_text(encoding="utf-8"))
    flat = []
    for pair in p["fuzzy_pairs"]:
        flat.append(norm(pair["a"]))
        flat.append(norm(pair["b"]))
    return flat


def norm_heading(h: str) -> str:
    """标题归一化：去掉（重复）类后缀与空白，用于标题覆盖检查。"""
    return re.sub(r"[（(](重复|重复标题)[）)]", "", h).replace(" ", "")


def quality(version: str, out_file: Path):
    """质量检查（与 consolidate.py --verify 同口径）：
    - 唯一内容丢失：既不在输出中、也不属于"已合并的近似对"的行数；
    - 近似重复未合并：一对的两侧都在输出中出现 → 未合并；
    - 标题缺失：含唯一内容的章节标题（归一化后）在输出中缺失的数量。
    """
    raw = DATA.read_text(encoding="utf-8")
    out_text = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
    out_norm = {norm(ln) for ln in out_text.splitlines() if norm(ln)}

    fuzzy = load_fuzzy_pairs()

    # 完全重复行（允许只保留一次）；内容行不含标题行
    lines = [ln for ln in raw.splitlines() if norm(ln) and not HEADING_RE.match(ln)]
    seen, exact_keys = set(), set()
    for ln in lines:
        k = norm(ln)
        if k in seen:
            exact_keys.add(k)
        seen.add(k)

    # 1) 唯一内容丢失（完全重复至少保留一处；近似对的任一侧保留即视为已合并）
    missing = []
    for ln in lines:
        k = norm(ln)
        if k in exact_keys:          # 完全重复：至少保留一处
            if k not in out_norm:
                missing.append(ln)
            continue
        if k in out_norm:
            continue
        if k in fuzzy:
            partner = fuzzy[fuzzy.index(k) ^ 1]
            if partner in out_norm:
                continue  # 该近似对已被合并（另一侧保留）
        missing.append(ln)

    # 2) 近似重复残留：两侧都在输出 → 未合并的对数
    fuzzy_residual = sum(1 for i in range(0, len(fuzzy), 2)
                         if fuzzy[i] in out_norm and fuzzy[i + 1] in out_norm)

    # 3) 标题覆盖（归一化后）：只要求"含非重复内容"的章节标题出现
    sections = {}
    cur = None
    for ln in raw.splitlines():
        m = HEADING_RE.match(ln)
        if m:
            cur = m.group(2).strip()
            sections.setdefault(cur, [])
        elif cur is not None and norm(ln):
            sections[cur].append(norm(ln))
    required = {norm_heading(h) for h, ls in sections.items()
                if any(k not in exact_keys and k not in fuzzy for k in ls)}
    heads_out = {norm_heading(m.group(2).strip())
                 for m in (HEADING_RE.match(l) for l in out_text.splitlines()) if m}
    miss_heads = required - heads_out

    return {
        "unique_content_lost": len(missing),
        "fuzzy_pairs_unmerged": fuzzy_residual,
        "headings_missing": len(miss_heads),
    }


def main():
    raw = DATA.read_text(encoding="utf-8")
    results = {"tokenizer": TOKENIZER, "llm_ms_per_1k_tokens_est": LLM_MS_PER_1K_TOKENS,
               "corpus": {"file": str(DATA), "chars": len(raw),
                          "tokens": count_tokens(raw)}, "versions": {}}

    # ---- 先跑两个预处理脚本（quality 需要 plan_v3 的近似重复对）----
    t2, log2 = run_script([sys.executable, str(SCRIPTS / "consolidate_v2.py"),
                           str(DATA), str(TMP / "plan_v2.json")], cwd=TMP)
    t3a, log3a = run_script([sys.executable, str(SCRIPTS / "consolidate.py"),
                             str(DATA), str(TMP / "plan_v3.json")], cwd=TMP)
    plan2 = (TMP / "plan_v2.json").read_text(encoding="utf-8")
    plan3 = (TMP / "plan_v3.json").read_text(encoding="utf-8")

    # ---- v1：纯 LLM 全量处理（无脚本）----
    out1 = OUTPUTS / "v1.md"
    s1 = (SKILL_DIR / "SKILL_v1.md").read_text(encoding="utf-8")
    results["versions"]["v1"] = {
        "skill": {"file": "SKILL_v1.md", "chars": len(s1), "tokens": count_tokens(s1)},
        "llm_input": {"source": "原文全文(raw_notes.md)",
                      "chars": len(raw), "tokens": count_tokens(raw)},
        "llm_output": {"file": "outputs/v1.md",
                       "chars": len(out1.read_text(encoding="utf-8")) if out1.exists() else 0,
                       "tokens": count_tokens(out1.read_text(encoding="utf-8")) if out1.exists() else 0},
        "script": {"cmd": "无（LLM 直接读全文）", "time_ms": 0.0},
        "quality": quality("v1", out1),
    }

    # ---- v2：脚本预处理（完全重复去重）----
    s2 = (SKILL_DIR / "SKILL_v2.md").read_text(encoding="utf-8")
    out2 = OUTPUTS / "v2.md"
    results["versions"]["v2"] = {
        "skill": {"file": "SKILL_v2.md", "chars": len(s2), "tokens": count_tokens(s2)},
        "llm_input": {"source": "plan.json(v2 预处理)",
                      "chars": len(plan2), "tokens": count_tokens(plan2)},
        "llm_output": {"file": "outputs/v2.md",
                       "chars": len(out2.read_text(encoding="utf-8")) if out2.exists() else 0,
                       "tokens": count_tokens(out2.read_text(encoding="utf-8")) if out2.exists() else 0},
        "script": {"cmd": "consolidate_v2.py", "time_ms": round(t2, 2), "log": log2},
        "quality": quality("v2", out2),
    }

    # ---- v3：自进化后（精简计划 + 模糊去重 + 决策式组装 + 无损校验）----
    s3 = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    dec3 = OUTPUTS / "decisions_v3.json"
    t3b, log3b = run_script([sys.executable, str(SCRIPTS / "consolidate.py"),
                             str(DATA), str(TMP / "plan_v3.json"),
                             "--assemble", str(dec3), "-o", str(OUTPUTS / "v3.md")], cwd=TMP)
    results["versions"]["v3"] = {
        "skill": {"file": "SKILL.md", "chars": len(s3), "tokens": count_tokens(s3)},
        "llm_input": {"source": "plan.json(v3 精简计划)",
                      "chars": len(plan3), "tokens": count_tokens(plan3)},
        "llm_output": {"file": "outputs/decisions_v3.json",
                       "chars": len(dec3.read_text(encoding="utf-8")) if dec3.exists() else 0,
                       "tokens": count_tokens(dec3.read_text(encoding="utf-8")) if dec3.exists() else 0},
        "script": {"cmd": "consolidate.py(plan + assemble)", "time_ms": round(t3a + t3b, 2),
                   "log": log3a + " | " + log3b},
        "quality": quality("v3", OUTPUTS / "v3.md"),
    }

    # ---- 汇总指标 ----
    for v in ("v1", "v2", "v3"):
        r = results["versions"][v]
        r["total_tokens_per_task"] = (r["skill"]["tokens"] + r["llm_input"]["tokens"]
                                      + r["llm_output"]["tokens"])
        r["llm_time_ms_est"] = round(r["total_tokens_per_task"] * LLM_MS_PER_1K_TOKENS / 1000)
        r["e2e_time_ms_est"] = round(r["llm_time_ms_est"] + r["script"]["time_ms"])

    out = WEEK14 / "bench" / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印摘要 ----
    print(f"tokenizer={TOKENIZER} | 语料 tokens={results['corpus']['tokens']}\n")
    hdr = f"{'版本':<5}{'SKILL(tok)':<12}{'输入(tok)':<12}{'输出(tok)':<12}" \
          f"{'合计(tok)':<12}{'脚本耗时':<12}{'LLM估算(ms)':<14}{'丢失':<6}{'近似重复未合并':<10}"
    print(hdr)
    for v in ("v1", "v2", "v3"):
        r = results["versions"][v]
        print(f"{v:<5}{r['skill']['tokens']:<12}{r['llm_input']['tokens']:<12}"
              f"{r['llm_output']['tokens']:<12}{r['total_tokens_per_task']:<12}"
              f"{r['script']['time_ms']:<12.2f}{r['llm_time_ms_est']:<14}"
              f"{r['quality']['unique_content_lost']:<6}{r['quality']['fuzzy_pairs_unmerged']:<10}")
    print("\nresults -> bench/results.json")


if __name__ == "__main__":
    main()
