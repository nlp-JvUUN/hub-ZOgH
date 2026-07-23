"""
构建向量索引（TF-IDF 版本）。
使用字符级 bigram + TF-IDF 进行文本向量化，余弦相似度检索。
DeepSeek 没有 embedding API，所以用纯本地方法实现向量化。
依赖: numpy, 标准库 json / pathlib
"""
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np

BASE_DIR = Path(__file__).parent.parent
CHUNK_PATH = BASE_DIR / "data" / "chunks" / "all_chunks.json"
VEC_DIR = BASE_DIR / "vectorstore"
VEC_DIR.mkdir(parents=True, exist_ok=True)

VEC_PATH = VEC_DIR / "vectors.npy"
VOCAB_PATH = VEC_DIR / "vocab.json"
META_PATH = VEC_DIR / "meta.json"


def tokenize(text: str) -> list[str]:
    """对中文文本做字符级 unigram + bigram 分词。"""
    chars = list(text)
    tokens = []
    for c in chars:
        if c.strip():
            tokens.append(c)
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if bigram.strip():
            tokens.append(bigram)
    return tokens


def build_vocab(chunks: list[dict], min_df: int = 1, max_df_ratio: float = 0.9) -> list[str]:
    """构建词汇表，过滤出现次数过少或过多的 term。"""
    n_docs = len(chunks)
    doc_freq = Counter()
    tokenized_docs = []

    for chunk in chunks:
        tokens = tokenize(chunk["content"])
        tokenized_docs.append(tokens)
        for token in set(tokens):
            doc_freq[token] += 1

    vocab = []
    for token, freq in doc_freq.items():
        if freq >= min_df and freq <= n_docs * max_df_ratio:
            vocab.append(token)

    print(f"词汇表大小: {len(vocab)} (从 {len(doc_freq)} 个候选词中筛选)")
    return vocab, tokenized_docs


def compute_tfidf(tokenized_docs: list[list[str]], vocab: list[str]) -> np.ndarray:
    """计算 TF-IDF 矩阵，L2 归一化。"""
    n_docs = len(tokenized_docs)
    vocab_size = len(vocab)
    term_to_idx = {t: i for i, t in enumerate(vocab)}

    # TF 矩阵
    tf_matrix = np.zeros((n_docs, vocab_size), dtype="float32")
    for i, tokens in enumerate(tokenized_docs):
        counter = Counter(tokens)
        max_freq = max(counter.values()) if counter else 1
        for term, count in counter.items():
            if term in term_to_idx:
                tf_matrix[i, term_to_idx[term]] = count / max_freq  # 归一化

    # IDF
    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1.0

    # TF-IDF
    tfidf = tf_matrix * idf

    # L2 归一化
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf = tfidf / np.maximum(norms, 1e-9)

    return tfidf


def main():
    print("=" * 60)
    print("构建向量索引 (TF-IDF)")
    print("=" * 60)

    # 加载 chunks
    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载 {len(chunks)} 个 chunks")

    # 构建词汇表
    vocab, tokenized_docs = build_vocab(chunks, min_df=1, max_df_ratio=0.9)

    # 计算 TF-IDF 向量
    vectors = compute_tfidf(tokenized_docs, vocab)
    print(f"向量矩阵: {vectors.shape}")

    # 保存
    np.save(VEC_PATH, vectors.astype("float32"))
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = vectors.nbytes / (1024 * 1024)
    print(f"向量已保存: {VEC_PATH} ({size_mb:.2f} MB)")
    print(f"词汇表已保存: {VOCAB_PATH}")


if __name__ == "__main__":
    main()
