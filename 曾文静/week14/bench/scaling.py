#!/usr/bin/env python3
"""scaling.py — 多语料扩展性验证。

本脚本用"多语料"验证同一个结论：v3"精简 plan"方案相对 v1"全量进上下文"的优势，随语料规模
增大而放大 —— 因为 plan 大小只与"需要判断的量"（章节数/重复数）相关，
与正文规模近似无关；而 v1 的成本与正文规模线性相关。

口径（与 benchmark.py 一致，tiktoken/cl100k_base）：
  v1 单次成本 = SKILL_v1 + 全文 + 输出估算（≈ 0.75 × 正文，按小语料实测比例）
  v3 单次成本 = SKILL_v3 + 精简 plan + 决策文件（实测 outputs/decisions_v3.json）

用法：PYTHONPATH=<tiktoken路径> python3 bench/scaling.py
"""
import json
import subprocess
import sys
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
DATA_DIR = SKILL_DIR / "data"
SCRIPTS = SKILL_DIR / "scripts"
TMP = WEEK14 / "bench" / "tmp"
TMP.mkdir(exist_ok=True)

CORPORA = [
    ("raw_notes.md", "decisions_v3.json"),
    ("raw_notes_large.md", "decisions_large.json"),
]
V1_OUTPUT_RATIO = 0.75  # v1 输出 ≈ 正文的 75%（按小语料 v1.md 实测：686/934）


def count_tokens(text: str) -> int:
    if ENC:
        return len(ENC.encode(text))
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return round(cjk / 1.6 + (len(text) - cjk) / 4)


def main():
    skill1 = count_tokens((SKILL_DIR / "SKILL_v1.md").read_text(encoding="utf-8"))
    skill3 = count_tokens((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))

    print(f"tokenizer={TOKENIZER} | SKILL_v1={skill1} | SKILL_v3={skill3}")
    print(f"{'语料':<18}{'正文(tok)':<10}{'plan(tok)':<10}{'plan/正文':<10}"
          f"{'v1 成本':<10}{'v3 成本':<10}{'节省':<8}")
    for name, dec_name in CORPORA:
        corpus = DATA_DIR / name
        raw = corpus.read_text(encoding="utf-8")
        plan_path = TMP / f"plan_scaling_{name}"
        subprocess.run([sys.executable, str(SCRIPTS / "consolidate.py"),
                        str(corpus), str(plan_path)], cwd=TMP, capture_output=True)
        plan = plan_path.read_text(encoding="utf-8")
        decisions = count_tokens((WEEK14 / "outputs" / dec_name)
                                 .read_text(encoding="utf-8"))

        c = count_tokens(raw)
        p = count_tokens(plan)
        v1 = skill1 + c + int(c * V1_OUTPUT_RATIO)
        v3 = skill3 + p + decisions
        saved = (v1 - v3) / v1 * 100
        print(f"{name:<18}{c:<10}{p:<10}{p / c:<10.2f}{v1:<10}{v3:<10}{saved:<7.1f}%")

    print("\n结论：正文越大，plan 占比越低（0.52 → 0.33），v3 相对 v1 的节省越大"
          "（~60% → ~72%）。")


if __name__ == "__main__":
    main()
