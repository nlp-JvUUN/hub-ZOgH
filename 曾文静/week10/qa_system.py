#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_system.py 第 2 步：检索 → 回答（核心逻辑，零第三方依赖）

流程（对应课件 Part 3/Part 4）：
  问题 → 分词 → BM25 打分 → 取 top-k → 相关性检查 → 回答

两种回答方式：
  * 保底模式（默认，无 API key）：直接返回最相关的原文片段——"能回答"的最小形态，
    且每段都带来源（课件页码），可溯源；
  * LLM 模式（设了 API key 后自动启用）：把 top-k 原文交给大模型，
    生成"只依据资料、标注来源编号、信息不足就明说"的答案（对应课件 Part 4 的 Prompt 设计）。

API 配置（任选其一）：
  export DASHSCOPE_API_KEY=sk-xxx                          # 默认 qwen-plus
  export OPENAI_API_KEY=sk-xxx
  export LLM_BASE_URL=https://api.deepseek.com             # 可换成 DeepSeek 等
  export LLM_MODEL=deepseek-chat

用法:
  python qa_system.py                          # 交互式问答
  python qa_system.py --query "什么是RAG"      # 单次提问
  python qa_system.py --query "..." --top-k 5  # 调整检索条数
  python qa_system.py --no-llm                 # 强制原文片段模式
  python qa_system.py --selftest               # 本地自测：9 道题 Hit@3 统计（不调 API）
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_PATH  = BASE_DIR / "data" / "knowledge.json"

TOP_K      = 3          # 送入回答的 chunk 数（课件建议 k=3~5）
NO_HIT     = "根据知识库未能找到与该问题相关的内容。"   # 防幻觉：检索不到就明说


# ─────────────────────────── 分词（无 jieba 的轻量方案） ───────────────────────────
def tokenize(text: str) -> list[str]:
    """英文/数字按整词，中文按单字+相邻二字组。够用于关键词检索。"""
    text = text.lower()
    toks = []
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
        if run[0].isascii():
            toks.append(run)
        else:
            toks.extend(run)                       # 单字
            if len(run) >= 2:
                toks.extend(run[i:i+2] for i in range(len(run) - 1))   # 二字组
    return toks


# ─────────────────────────── BM25（自实现，约 40 行） ───────────────────────────
class BM25:
    """Okapi BM25：词频(TF) + 逆文档频率(IDF) + 文档长度归一化。对应课件 Part 3。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus     = corpus
        self.doc_len    = [len(d) for d in corpus]
        self.avgdl      = sum(self.doc_len) / max(len(corpus), 1)
        self.N          = len(corpus)
        df = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], doc: list[str]) -> float:
        dl  = len(doc)
        tf  = Counter(doc)
        s   = 0.0
        for t in set(query):
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ─────────────────────────── 知识库加载 ───────────────────────────
class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.chunks = data["chunks"]
        self.bm25   = BM25([tokenize(c["content"]) for c in self.chunks])
        src = data.get("source_files") or [data.get("source_file", "")]
        print(f"[INFO] 知识库加载完成: {len(self.chunks)} 块 ← {', '.join(Path(s).name for s in src)}")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q_tokens = tokenize(query)
        # 有效词 = 二字组/英文词（长度>=2）。单字参与度低且噪声大（如"入"既匹配
        # "收入"也匹配"输入"），只保留有效词做相关性判定与打分 —— 这也是
        # "检索不到相关内容就拒绝回答"（防幻觉第一道闸门）的判定依据。
        sig = [t for t in q_tokens if len(t) >= 2] or q_tokens
        scored = []
        for c in self.chunks:
            dt = set(tokenize(c["content"]))
            # 相关性门槛：至少命中一个"非纯数字"的有效词。
            # 纯数字（如"2023"）在论文参考文献里到处都是，单独命中不算相关，
            # 否则"茅台2023年营收"会因年份噪声被误判为命中（防幻觉第一道闸门）。
            if not any(t in dt and not t.isdigit() for t in sig):
                continue
            scored.append((self.bm25.score(sig, tokenize(c["content"])), c))
        scored.sort(key=lambda x: -x[0])
        hits = [dict(c, score=round(s, 4)) for s, c in scored[:top_k]]
        return hits, q_tokens


# ─────────────────────────── 回答：原文片段（保底） ───────────────────────────
def answer_by_snippet(hits: list[dict]) -> str:
    if not hits:
        return NO_HIT
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['source']}\n{h['content']}")
    return "\n\n".join(lines)


# ─────────────────────────── 回答：LLM 生成（可选） ───────────────────────────
SYSTEM_PROMPT = """你是一个知识助手，根据【参考资料】回答用户问题。
回答规则：
1. 只根据参考资料中的内容回答，不得编造资料外的内容；
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"；
3. 引用具体内容时在句末标注来源编号，如：...（[1]）；
4. 回答简洁，重点突出。"""


def llm_available() -> bool:
    return bool(__import__("os").getenv("DASHSCOPE_API_KEY") or __import__("os").getenv("OPENAI_API_KEY"))


def answer_by_llm(question: str, hits: list[dict]) -> str:
    if not hits:
        return NO_HIT
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    context = "\n\n---\n\n".join(f"[{i}] {h['source']}\n{h['content']}" for i, h in enumerate(hits, 1))
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{question}"},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ─────────────────────────── 主流程 ───────────────────────────
def ask(kb: KnowledgeBase, question: str, top_k: int, use_llm: bool, verbose: bool = False) -> dict:
    hits, q_tokens = kb.retrieve(question, top_k)
    if verbose:
        print(f"[INFO] 检索: 命中 {len(hits)} 块 | 最高分 {hits[0]['score'] if hits else 0} | 分词: {'/'.join(q_tokens[:12])}")

    if use_llm:
        try:
            answer = answer_by_llm(question, hits)
        except Exception as e:
            print(f"[WARN] LLM 调用失败（{e}），降级为原文片段模式")
            answer = answer_by_snippet(hits)
            use_llm = False
    else:
        answer = answer_by_snippet(hits)

    return {
        "question": question,
        "answer": answer,
        "citations": [{"source": h["source"], "score": h["score"]} for h in hits],
        "mode": "llm" if use_llm else "snippet",
    }


def print_result(r: dict):
    print(f"\n{'=' * 64}\n问题：{r['question']}\n{'=' * 64}")
    print(r["answer"])
    if r["citations"]:
        print("\n── 来源 ──")
        for c in r["citations"]:
            print(f"  {c['source']}  (score={c['score']})")
    print(f"[模式: {r['mode']}]")


# ─────────────────────────── 自测：9 道题 Hit@3（不调 API） ───────────────────────────
SELFTEST = [
    # (问题, 期望来源页, 说明)
    ("什么是RAG？RAG的三个核心步骤是什么？",        "[RAG.pptx] 第6页",  "基础概念"),
    ("大模型为什么会产生幻觉？",                     "[RAG.pptx] 第4页",  "动机理解"),
    ("文本分块时chunk overlap有什么作用？",          "[RAG.pptx] 第11页", "分块细节"),
    ("BM25打分主要考虑哪三个因素？",                 "[RAG.pptx] 第18页", "检索原理"),
    ("RRF混合检索的公式是什么？",                    "[RAG.pptx] 第19页", "混合检索"),
    ("RAGAS中Faithfulness指标衡量什么？",            "[RAG.pptx] 第30页", "评估指标"),
    ("RAGAS框架是什么？怎么用？",                    "[RAG.pptx] 第33页", "评估框架"),
    ("课件推荐的中文Embedding模型有哪些？",          "[RAG.pptx] 第13页", "选型知识"),
    ("什么是Graph RAG？适合什么场景？",              "[RAG.pptx] 第40页", "进阶架构"),
    ("贵州茅台2023年营业收入是多少？",               None,                "越界问题（期望拒绝）"),
]


def selftest(kb: KnowledgeBase, top_k: int = 3):
    print(f"\n{'=' * 64}\n自测：{len(SELFTEST)} 道题，Hit@{top_k}（命中=期望来源页出现在检索结果中）\n{'=' * 64}")
    n_hit = n_refuse_ok = 0
    for q, expect, tag in SELFTEST:
        hits, _ = kb.retrieve(q, top_k)
        sources = [h["source"] for h in hits]
        if expect is None:                                   # 越界问题：期望拒绝
            ok = not sources
            n_refuse_ok += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} | {q}")
            print(f"     期望: 拒绝回答 | 实际: {'拒绝' if ok else '检索到了内容(未拒绝)'} | top: {sources[:2]}")
            continue
        ok = expect in sources
        n_hit += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {tag} | {q}")
        print(f"     期望: {expect} | 实际 top-{top_k}: {sources}")
    print(f"\n结果: Hit@{top_k} = {n_hit}/{len(SELFTEST) - 1}  |  越界拒绝 = {n_refuse_ok}/1")


# ─────────────────────────── 入口 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="基于手头文件的问答系统（最小闭环）")
    ap.add_argument("--query", type=str, default=None, help="单次提问")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="检索条数")
    ap.add_argument("--no-llm", action="store_true", help="强制原文片段模式")
    ap.add_argument("--llm", action="store_true", help="强制 LLM 模式（需 API key）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="本地自测")
    args = ap.parse_args()

    kb = KnowledgeBase()
    use_llm = (args.llm or llm_available()) and not args.no_llm
    print(f"[INFO] 回答模式: {'LLM 生成' if use_llm else '原文片段（保底）'}"
          f" | top-k={args.top_k} | 提示: 设 DASHSCOPE_API_KEY 可启用 LLM 模式")

    if args.selftest:
        selftest(kb, args.top_k)
        return

    if args.query:
        print_result(ask(kb, args.query, args.top_k, use_llm, args.verbose))
        return

    print("\n问答系统（输入 exit 退出）")
    while True:
        try:
            q = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        print_result(ask(kb, q, args.top_k, use_llm, args.verbose))


if __name__ == "__main__":
    main()
