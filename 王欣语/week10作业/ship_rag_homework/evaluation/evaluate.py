"""
RAG 系统评估脚本（船舶术语版）

对 RAG 系统进行标准化评测，输出核心指标：
  - Hit Rate @4：前4个召回结果中是否包含目标文档
  - MRR（Mean Reciprocal Rank）：第一个命中的平均倒数排名
  - 答案匹配度：与 ground truth 的文本相似度

使用方式：
  python evaluate.py                    # 评估全部20题
  python evaluate.py --question-ids 1,2,3  # 只跑部分题
  python evaluate.py --skip-llm         # 跳过 LLM 生成，只测检索指标

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
EVAL_DIR   = Path(__file__).parent
RESULT_DIR = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# 将 src 加入路径
sys.path.insert(0, str(BASE_DIR / "src"))


# ── 加载评测题集 ──────────────────────────────────────────────────────────────

def load_questions(question_ids: list[int] = None) -> list[dict]:
    qpath = EVAL_DIR / "questions.json"
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    return questions


# ── RAG 调用 ──────────────────────────────────────────────────────────────────

def run_rag_eval(questions: list[dict], skip_llm: bool = False) -> list[dict]:
    """
    调用 src/rag_pipeline.py 中的 RAGPipeline 跑所有测试题。
    返回评估所需格式：[{question, answer, contexts, ground_truth, retrieved}]
    """
    from rag_pipeline import RAGPipeline

    logger.info("初始化 RAG Pipeline...")
    pipeline = RAGPipeline(use_bm25=True, use_rerank=False)  # 评估时关闭 Rerank 加快速度

    results = []
    for q in questions:
        logger.info(f"[{q['id']:02d}] {q['question'][:40]}...")
        try:
            result = pipeline.query(q["question"], verbose=False)
            results.append({
                "question":     q["question"],
                "answer":       result["answer"],
                "contexts":     [r["content"] for r in result.get("retrieved", [])],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
                "target_docs":  q.get("target_docs", []),
                "retrieved":    result.get("retrieved", []),
            })
        except Exception as e:
            logger.error(f"  题目 {q['id']} 失败: {e}")
            results.append({
                "question":     q["question"],
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": q["ground_truth"],
                "question_id":  q["id"],
                "question_type":q["type"],
                "target_docs":  q.get("target_docs", []),
                "retrieved":    [],
            })

    return results


# ── 检索指标计算 ───────────────────────────────────────────────────────────────

def compute_retrieval_metrics(results: list[dict], top_k: int = 4) -> dict:
    """
    计算检索层面指标：
      - Hit Rate @k：目标文档是否出现在 top-k 中
      - MRR @k：平均倒数排名
    """
    hits = []
    mrrs = []

    for r in results:
        target_docs = r.get("target_docs", [])
        if not target_docs:
            continue  # should_refuse 类跳过

        retrieved = r.get("retrieved", [])[:top_k]

        hit_any = False
        first_rank = None
        for i, item in enumerate(retrieved, 1):
            source_file = item.get("source_file", "")
            # 检查是否匹配目标文档
            for td in target_docs:
                if td in source_file:
                    hit_any = True
                    if first_rank is None:
                        first_rank = i
                    break
            if hit_any and first_rank:
                break

        hits.append(1 if hit_any else 0)
        mrrs.append(1 / first_rank if first_rank else 0)

    return {
        "hit_rate": np.mean(hits) if hits else 0.0,
        "mrr": np.mean(mrrs) if mrrs else 0.0,
        "n_questions": len(hits),
    }


# ── 答案匹配度（简单版）────────────────────────────────────────────────────────

def compute_answer_metrics(results: list[dict]) -> dict:
    """
    简单计算答案与 ground truth 的匹配度：
      - 包含关键信息的题数（ground_truth 中的关键词出现在 answer 中）
    """
    matched = 0
    total = 0

    for r in results:
        if r.get("question_type") == "should_refuse":
            # 检查是否正确拒绝
            refused = any(kw in r["answer"] for kw in ["无法回答", "无法提供", "不在", "不包含", "超出"])
            if refused:
                matched += 1
            total += 1
            continue

        gt = r.get("ground_truth", "")
        ans = r.get("answer", "")

        # 简单判断：ground truth 中长度 > 4 的词是否出现在 answer 中
        import jieba
        gt_words = set(w for w in jieba.cut(gt) if len(w) >= 4)
        ans_words = set(w for w in jieba.cut(ans) if len(w) >= 4)

        if gt_words and len(gt_words & ans_words) >= min(2, len(gt_words)):
            matched += 1
        total += 1

    return {
        "answer_match_rate": matched / total if total else 0.0,
        "n_questions": total,
    }


# ── 按题型分析 ────────────────────────────────────────────────────────────────

def analyze_by_type(results: list[dict]):
    """统计不同题型的基本表现。"""
    from collections import defaultdict

    type_stats = defaultdict(list)
    for r in results:
        qtype  = r.get("question_type", "unknown")
        answer = r.get("answer", "")
        refused = any(kw in answer for kw in ["无法回答", "无法提供", "不在", "不包含", "超出"])
        type_stats[qtype].append({"refused": refused, "answer_len": len(answer)})

    print("\n── 按题型统计 ──")
    for qtype, stats in type_stats.items():
        refuse_rate = sum(1 for s in stats if s["refused"]) / len(stats)
        avg_len     = sum(s["answer_len"] for s in stats) / len(stats)
        print(f"  {qtype:<25} 题数={len(stats)}  拒绝率={refuse_rate:.0%}  平均回答长度={avg_len:.0f}字")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(raw_results: list[dict], ret_metrics: dict, ans_metrics: dict):
    import pandas as pd
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULT_DIR / f"eval_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "retrieval_metrics": ret_metrics,
        "answer_metrics": ans_metrics,
        "raw_results": raw_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存 → {out_path}")

    # 生成 CSV 摘要
    df = pd.DataFrame([{
        "id":     r["question_id"],
        "type":   r["question_type"],
        "question": r["question"][:30] + "...",
        "answer_len": len(r["answer"]),
        "context_count": len(r["contexts"]),
    } for r in raw_results])
    csv_path = RESULT_DIR / f"eval_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 摘要已保存 → {csv_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="船舶术语 RAG 系统评估")
    parser.add_argument("--question-ids", type=str, default=None, help="指定题号，逗号分隔，如 1,2,3")
    parser.add_argument("--skip-llm",     action="store_true", help="跳过 LLM 生成，只测检索")
    args = parser.parse_args()

    q_ids = [int(x) for x in args.question_ids.split(",")] if args.question_ids else None
    questions = load_questions(q_ids)
    logger.info(f"加载 {len(questions)} 道测试题")

    # 运行 RAG
    results = run_rag_eval(questions, skip_llm=args.skip_llm)
    analyze_by_type(results)

    # 计算指标
    ret_metrics = compute_retrieval_metrics(results)
    ans_metrics = compute_answer_metrics(results)

    print(f"\n{'='*50}")
    print(f"评估结果")
    print(f"{'='*50}")
    print(f"  检索指标（Top-4）：")
    print(f"    Hit Rate: {ret_metrics['hit_rate']:.3f} ({ret_metrics['n_questions']} 题)")
    print(f"    MRR:      {ret_metrics['mrr']:.3f}")
    print(f"  答案指标：")
    print(f"    匹配率:   {ans_metrics['answer_match_rate']:.3f} ({ans_metrics['n_questions']} 题)")
    print(f"{'='*50}")

    # 保存结果
    save_results(results, ret_metrics, ans_metrics)


if __name__ == "__main__":
    main()
