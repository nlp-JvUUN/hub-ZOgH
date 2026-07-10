"""
RAG 问答系统

流程：
  向量检索（本地 bge-small-zh + FAISS）
      +
  BM25 关键词检索（jieba + rank_bm25）
      ↓
  RRF 融合排名
      ↓
  LLM 生成（DashScope / OpenAI 兼容接口）+ 引用标注

用法：
  python rag_qa.py                              # 交互式问答
  python rag_qa.py --query "茅台2023年营收"      # 单次提问
  python rag_qa.py --query "..." --stock 600519  # 指定股票筛选

环境变量：
  DASHSCOPE_API_KEY  — 阿里云 DashScope API Key（用于 LLM 生成）
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
WEEK10_DIR = BASE_DIR.parent
VEC_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VEC_DIR / "faiss_index.bin"
META_PATH = VEC_DIR / "faiss_meta.json"

# 本地 embedding 模型
EMBED_MODEL_PATH = WEEK10_DIR / "rag_annual_report" / "models" / "bge-small-zh-v1.5"

# ── LLM 配置（按需修改） ─────────────────────────────────────────────────────
# 兼容 OpenAI 接口格式，可对接 DashScope / OpenAI / DeepSeek / Ollama 等
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY_ENV = "DASHSCOPE_API_KEY"   # 从环境变量读取 API Key
LLM_MODEL = "qwen3.7-plus"                 # 可换 qwen-turbo / qwen-max / gpt-4o 等
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_RETRIEVE = 10
TOP_K_FINAL = 4
SCORE_THRESHOLD = 0.25

SYSTEM_PROMPT = """你是一个专业的测试用例知识问答助手，能够根据提供的测试用例资料回答用户的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体内容时，在句末标注来源编号，如：xxx[1]
4. 涉及步骤、预期结果等内容要完整准确，不得遗漏或简化
5. 回答简洁，重点突出，避免无关废话"""


# ── LLM 客户端 ────────────────────────────────────────────────────────────────

def get_llm_client():
    """
    获取 LLM 客户端（OpenAI 兼容接口）。
    可对接 DashScope / OpenAI / DeepSeek / Ollama 等。
    """
    from openai import OpenAI

    api_key = os.getenv(LLM_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"请设置环境变量 {LLM_API_KEY_ENV}，例如：\n"
            f'  export {LLM_API_KEY_ENV}="sk-xxx"'
        )
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


def call_llm(query: str, context: str, client) -> str:
    """调用 LLM 生成回答。"""
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ── 向量检索 ──────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        self.meta_list = meta["chunks"]
        embed_dim = meta["embed_dim"]

        self.index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

        logger.info(f"加载本地 embedding 模型: {EMBED_MODEL_PATH}")
        self.model = SentenceTransformer(str(EMBED_MODEL_PATH))

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE,
               filter_meta: Optional[dict] = None) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)
            if filter_meta:
                if not all(str(item.get(k, "")) == str(v) for k, v in filter_meta.items()):
                    continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results


# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

class BM25Store:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        self.meta_list = meta["chunks"]

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 索引构建完成")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(vec_results: list[dict], bm25_results: list[dict],
                           k: int = 60) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    return [dict(chunk_map[cid]) | {"rrf_score": rrf_scores[cid]} for cid in sorted_cids]


# ── 上下文组装 ────────────────────────────────────────────────────────────────

def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    citations = []
    for i, item in enumerate(retrieved, 1):
        source = item.get("source", "")
        case_id = item.get("case_id", "")
        module = item.get("module", "")

        label = f"[{i}] {source}"
        if case_id:
            label += f" · {case_id}"
        if module:
            label += f" · {module}"

        parts.append(f"{label}\n{item['content']}")
        citations.append({"index": i, "source": label})

    return "\n\n---\n\n".join(parts), citations


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, use_bm25: bool = True):
        self.vec_store = VectorStore()
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Store() if use_bm25 else None
        self.llm_client = get_llm_client()

    def query(self, question: str, filter_meta: Optional[dict] = None,
              verbose: bool = False) -> dict:
        # ① 向量检索
        vec_results = self.vec_store.search(question, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            n = len(vec_results)
            s = vec_results[0]["vec_score"] if vec_results else 0
            logger.info(f"向量召回: {n} 条，最高分={s:.3f}")

        # ② BM25 + RRF 融合
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, TOP_K_RETRIEVE)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ③ 取 top-K
        final = candidates[:TOP_K_FINAL]
        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        # ④ 相关性阈值检查
        if not final:
            return {"answer": "未找到相关内容，无法回答此问题。", "citations": [], "retrieved": []}

        top_score = final[0].get("vec_score", 1.0)
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据知识库未能找到与该问题相关的内容，建议查阅原始文件。",
                "citations": [], "retrieved": final,
            }

        # ⑤ LLM 生成
        context, citations = build_context(final)
        answer = call_llm(question, context, self.llm_client)
        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="week10 RAG 问答系统")
    parser.add_argument("--query", type=str, default=None, help="单次提问内容")
    parser.add_argument("--module", type=str, default=None, help="按所属模块筛选")
    parser.add_argument("--priority", type=str, default=None, help="按用例等级筛选，如 P1")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 检索")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        logger.error(f"索引文件不存在: {INDEX_PATH}")
        logger.error("请先运行 python build_index.py 构建索引")
        return

    pipeline = RAGPipeline(use_bm25=not args.no_bm25)

    filter_meta = {}
    if args.module:
        filter_meta["module"] = args.module
    if args.priority:
        filter_meta["priority"] = args.priority
    filter_meta = filter_meta or None

    def print_result(q: str, result: dict):
        print(f"\n{'=' * 60}")
        print(f"问题：{q}")
        print(f"{'=' * 60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 参考来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("=" * 60)
        print("  测试用例 RAG 问答系统")
        print(f"  LLM: {LLM_MODEL}  |  Embedding: bge-small-zh-v1.5 (本地)")
        print("  数据源: test_cases.xlsx (138 条测试用例)")
        print("  输入 'exit' 退出")
        print("=" * 60)
        print()
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()
