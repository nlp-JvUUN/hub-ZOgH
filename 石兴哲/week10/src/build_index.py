"""
向量索引构建脚本（原生实现）

Embedding 方案：阿里云 DashScope text-embedding-v3
  - 维度：1024
  - 每批最多 10 条（API 硬限制）

向量库：FAISS IndexFlatIP（内积 = 归一化后的余弦相似度）

与 week10_rag 的差异：
  - 元数据字段: patent_id / title / assignee / patent_office
  - 权利要求块的元数据中包含 claim_num（用于过滤独立权利要求）
  - 其他逻辑完全一致——证明 RAG 管道的"数据获取/解析/分块/建索引"分层设计通用性好

依赖：
  pip install faiss-cpu openai numpy
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL     = "text-embedding-v3"
EMBED_DIM       = 1024
BATCH_SIZE      = 10
DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
            "  Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    """批量计算 embedding，返回 shape=(N, 1024) 的 L2 归一化数组。"""
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 100 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                all_embeddings.extend([e.embedding for e in resp.data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], client: OpenAI):
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(client, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":      c["chunk_id"],
            "content":       c["content"],
            "patent_id":     c["metadata"].get("patent_id", ""),
            "title":         c["metadata"].get("title", ""),
            "assignee":      c["metadata"].get("assignee", ""),
            "patent_office": c["metadata"].get("patent_office", ""),
            "section":       c["metadata"].get("section", ""),
            "block_types":   c["metadata"].get("block_types", []),
            "claim_num":     c["metadata"].get("claim_num", 0),
            "strategy":      c["metadata"].get("strategy", ""),
            "source_file":   c["metadata"].get("source_file", ""),
            "parent_content":c["metadata"].get("parent_content", ""),
            "parent_id":     c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    client = get_client()
    build_faiss_index(chunks, client)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
