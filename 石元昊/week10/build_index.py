"""
解析 test_cases.xlsx 测试用例文件，构建 FAISS 向量索引。

用法：
  python build_index.py

输出：
  vectorstore/faiss_index.bin
  vectorstore/faiss_meta.json
"""

import json
import logging
import hashlib
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
WEEK10_DIR = BASE_DIR.parent
VEC_DIR = BASE_DIR / "vectorstore"

# 本地 embedding 模型（已存在于 rag_annual_report/models/）
EMBED_MODEL_PATH = WEEK10_DIR / "rag_annual_report" / "models" / "bge-small-zh-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ── 文件解析 ──────────────────────────────────────────────────────────────────

def parse_xlsx(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_excel(path)
    docs = []
    for _, row in df.iterrows():
        case_id = str(row.get("ID", "")).strip()
        case_name = str(row.get("用例名称", "")).strip()
        precondition = str(row.get("前置条件", "")).strip()
        module = str(row.get("所属模块", "")).strip()
        steps = str(row.get("步骤描述", "")).strip()
        expected = str(row.get("预期结果", "")).strip()
        tags = str(row.get("标签", "")).strip()
        priority = str(row.get("用例等级", "")).strip()
        related_task = str(row.get("关联任务", "")).strip()

        text_parts = [
            f"用例名称：{case_name}",
            f"用例ID：{case_id}",
            f"所属模块：{module}",
            f"用例等级：{priority}",
        ]
        if precondition and precondition != "nan":
            text_parts.append(f"前置条件：{precondition}")
        if steps and steps != "nan":
            text_parts.append(f"步骤描述：{steps}")
        if expected and expected != "nan":
            text_parts.append(f"预期结果：{expected}")
        if tags and tags != "nan":
            text_parts.append(f"标签：{tags}")
        if related_task and related_task != "nan":
            text_parts.append(f"关联任务：{related_task}")

        docs.append({
            "source": path.name,
            "source_type": "xlsx",
            "page": -1,
            "text": "\n".join(text_parts),
            "case_id": case_id,
            "module": module,
            "priority": priority,
            "tags": tags if tags != "nan" else "",
        })
    return docs


def collect_documents() -> list[dict]:
    docs = []

    xlsx_path = BASE_DIR / "data" / "test_cases.xlsx"
    if xlsx_path.exists():
        logger.info(f"解析 XLSX: {xlsx_path.name}")
        docs = parse_xlsx(xlsx_path)
    else:
        logger.error(f"文件不存在: {xlsx_path}")

    logger.info(f"共解析 {len(docs)} 条记录")
    return docs


# ── 文本分块 ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = hashlib.md5(f"{doc['source']}_{doc.get('case_id', '')}_{i}".encode()).hexdigest()[:12]
            chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "source_type": doc["source_type"],
                "page": doc.get("page", -1),
                "chunk_index": i,
                "content": chunk,
                "case_id": doc.get("case_id", ""),
                "module": doc.get("module", ""),
                "priority": doc.get("priority", ""),
                "tags": doc.get("tags", ""),
            })
    logger.info(f"共分块 {len(chunks)} 条")
    return chunks


# ── 向量索引 ──────────────────────────────────────────────────────────────────

def build_vector_index(chunks: list[dict]):
    import faiss
    from sentence_transformers import SentenceTransformer

    VEC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载本地 embedding 模型: {EMBED_MODEL_PATH}")
    model = SentenceTransformer(str(EMBED_MODEL_PATH))

    contents = [c["content"] for c in chunks]
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(vecs)
        logger.info(f"Embedding 进度: {min(i + batch_size, len(contents))}/{len(contents)}")

    embeddings = np.vstack(all_embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(VEC_DIR / "faiss_index.bin"))

    # 保存维度信息供 rag_qa.py 使用
    meta = {"embed_dim": dim, "chunks": chunks}
    with open(VEC_DIR / "faiss_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    logger.info(f"FAISS 索引已保存至 {VEC_DIR}（维度={dim}，共 {index.ntotal} 条）")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    docs = collect_documents()
    if not docs:
        logger.error("未找到任何文件，请检查路径")
        return
    chunks = build_chunks(docs)
    build_vector_index(chunks)
    logger.info("索引构建完成！")


if __name__ == "__main__":
    main()
