"""
向量索引构建脚本（LangChain 版 - 船舶术语）

与原生版（src/build_index.py）的对比：
  原生版：DashScope API embedding + 手动 FAISS 操作
  本版本：本地 BGE 模型 + LangChain 封装，完全离线运行

Embedding：本地 BAAI/bge-small-zh-v1.5（512维）
  路径：/Users/wangxinyu/Desktop/python/最新/pretrain_models/bge-small-zh-v1.5

向量库：LangChain FAISS 封装
  保存路径：vectorstore/faiss_lc/

依赖：
  pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODEL_PATH      = "/Users/wangxinyu/Desktop/python/最新/pretrain_models/bge-small-zh-v1.5"

STRATEGY        = "semantic"
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"


# ── 1. 加载 chunks（从 JSON 转为 LangChain Document）───────────────────────────

def load_documents():
    """将之前生成的 chunks JSON 转为 LangChain Document 列表。"""
    from langchain_core.documents import Document

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"找不到 {CHUNKS_FILE}，请先运行 src/chunk_documents.py")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for c in chunks:
        docs.append(Document(
            page_content=c["content"],
            metadata={
                "chunk_id":   c["chunk_id"],
                "doc_source": c.get("metadata", {}).get("doc_source", ""),
                "category":   c.get("metadata", {}).get("category", ""),
                "row_num":    c.get("metadata", {}).get("row_num", -1),
                "strategy":   c.get("metadata", {}).get("strategy", ""),
            }
        ))

    logger.info(f"加载 {len(docs)} 个 chunks")
    return docs


# ── 2. Embedding 模型（本地 BGE）──────────────────────────────────────────────

def get_embeddings():
    """加载本地 BGE 模型。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"本地模型不存在: {MODEL_PATH}\n"
            "请先运行: python download_bge.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding 模型加载完成: {MODEL_PATH}")
    return embeddings


# ── 3. 构建并保存 FAISS 向量库 ────────────────────────────────────────────────

def build_vectorstore(docs, embeddings):
    """LangChain 一行构建 FAISS 向量库。"""
    from langchain_community.vectorstores import FAISS

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建向量库（{len(docs)} 个 chunk）...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"向量库已保存 → {VECTORSTORE_DIR}")
    return vectorstore


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # 步骤 1: 加载
    docs = load_documents()

    # 步骤 2: Embedding
    embeddings = get_embeddings()

    # 步骤 3: 建库
    build_vectorstore(docs, embeddings)

    print(f"\nLangChain 向量库构建完成！")
    print(f"  路径: {VECTORSTORE_DIR}")
    print(f"  下一步: python src_langchain/rag_chain_lc.py")


if __name__ == "__main__":
    main()
