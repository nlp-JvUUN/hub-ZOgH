"""
向量索引构建脚本（本地 Embedding 实现）

Embedding 方案：本地 BAAI/bge-small-zh-v1.5
  - 模型已下载到项目 models/ 目录，无需联网
  - 维度：512（bge-small-zh 的固定维度）
  - 免费、离线可用，适合教学场景
  - 中文检索效果优秀

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

依赖：
  pip install faiss-cpu sentence-transformers numpy

说明：
  LLM 生成仍使用 DashScope qwen-plus（需设置 DASHSCOPE_API_KEY），
  但 Embedding 全部走本地模型，零成本、可离线。
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"          # 与 chunk_documents.py 保持一致
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

# 本地 Embedding 模型配置
LOCAL_MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"
EMBED_DIM       = 512                 # bge-small-zh-v1.5 固定输出维度
BATCH_SIZE      = 32                  # 本地模型批量大小可设大一些


# ── 本地 Embedding 模型 ────────────────────────────────────────────────────────

_embedding_model = None   # 全局缓存，避免重复加载

class _TransformersEmbedder:
    """sentence-transformers 不可用时的兜底实现：直接用 transformers 加载 BERT。"""
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self._dim = self.model.config.hidden_size

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, batch_size=32, normalize_embeddings=False,
               show_progress_bar=False, convert_to_numpy=True):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")
            with self.torch.no_grad():
                outputs = self.model(**encoded)
            # mean pooling over token embeddings (attention-masked)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            vecs = summed / counts
            all_vecs.append(vecs)
        embeddings = self.torch.cat(all_vecs, dim=0).cpu().numpy().astype("float32")
        return embeddings

def get_embedder():
    """懒加载本地 Embedding 模型（bge-small-zh-v1.5）。
    优先用 sentence-transformers；不可用时回退到 transformers 原生实现。"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5"
    if not LOCAL_MODEL_DIR.exists():
        logger.warning(f"本地模型目录不存在: {LOCAL_MODEL_DIR}，尝试在线下载 BAAI/bge-small-zh-v1.5")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载本地 Embedding 模型（sentence-transformers）: {model_path}")
        _embedding_model = SentenceTransformer(model_path)
    except ImportError:
        logger.info("sentence-transformers 未安装，回退到 transformers 原生实现")
        _embedding_model = _TransformersEmbedder(model_path)

    logger.info(f"模型加载完成，输出维度={_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


# ── Embedding 计算 ────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    批量计算 embedding。
    返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。

    bge 模型官方建议：对检索 query 加前缀 "为这个句子生成表示以用于检索相关文章："
    此处文档侧不加前缀（仅 query 侧加），保持一致性。
    """
    model = get_embedder()

    # show_progress_bar 在终端可关闭以免刷屏
    embeddings = model.encode(
        texts,
        batch_size     = BATCH_SIZE,
        normalize_embeddings = False,   # 手动归一化，与原实现保持一致
        show_progress_bar   = show_progress and len(texts) > 100,
        convert_to_numpy    = True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # L2 归一化（使内积等于余弦相似度）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)     # 防止除零
    embeddings = embeddings / norms

    return embeddings


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict]):
    """
    构建 FAISS 向量索引。

    FAISS 说明：
      IndexFlatIP = 暴力内积检索，精确但不近似。
      数据量 < 10 万时速度完全够用，是教学的首选。
      数据量更大时可换 IndexIVFFlat（需要 train）或 IndexHNSW。
    """
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化：索引文件 + 元数据（分开存，避免把大向量序列化进 JSON）
    # 注意：faiss 的 C++ IO 在 Windows 上不支持非 ASCII 路径（如中文目录名），
    # 因此用 serialize_index 序列化为字节，再用 Python 文件 IO 写入（支持任意路径）。
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    index_bytes = faiss.serialize_index(index)   # 返回 numpy array
    with open(index_path, "wb") as f:
        f.write(index_bytes.tobytes())
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":   c["chunk_id"],
            "content":    c["content"],
            "title":      c["metadata"].get("title", ""),
            "topic":      c["metadata"].get("topic", ""),
            "page_num":   c["metadata"].get("page_num", -1),
            "section":    c["metadata"].get("section", ""),
            "block_types":c["metadata"].get("block_types", []),
            "is_ocr":     c["metadata"].get("is_ocr", False),
            "strategy":   c["metadata"].get("strategy", ""),
            "source_file":c["metadata"].get("source_file", ""),
            # 层级分块时保留父块内容供 LLM 读取
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id":      c["metadata"].get("parent_id", ""),
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

    build_faiss_index(chunks)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
