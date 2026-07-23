"""Fast retrieval evaluation using manually labelled relevant annual-report pages.

No answer generation or LLM-as-a-judge is used: one query embedding per method,
then Recall@4, MRR@4, and latency are calculated from the returned source pages.
"""
import csv
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import API_KEY, BASE_URL, VECTOR_DIR, RESULTS_DIR, EMBED_MODEL

# Gold pages were manually labelled from the extracted BYD 2023 annual report.
# A range is used where one answer naturally spans a page break.
TESTS = [
    {"question": "2023 年营业收入是多少，同比增长多少？", "gold_pages": {11}},
    {"question": "汽车及相关产品的收入和同比增速是多少？", "gold_pages": {20}},
    {"question": "公司前五大客户销售额占比是多少？", "gold_pages": {22}},
    {"question": "2023 年公司研发投入与研发人员变化情况如何？", "gold_pages": {27, 28}},
    {"question": "比亚迪新能源乘用车进入了多少国家和地区？", "gold_pages": {16}},
    {"question": "公司对 2024 年的主要发展展望是什么？", "gold_pages": {35, 36, 37}},
    {"question": "报告期末员工总数是多少？", "gold_pages": {63}},
    {"question": "生产废水是如何处理的？", "gold_pages": {90}},
    {"question": "审计报告出具的审计意见类型是什么？", "gold_pages": {132}},
    {"question": "经营活动产生的现金流量表位于哪一页？", "gold_pages": {144}},
]


def bm25_docs(query, data, k=4):
    tokens = list(re.sub(r"\s+", "", query))
    scores = data["bm25"].get_scores(tokens)
    return [data["chunks"][i] for i in scores.argsort()[::-1][:k]]


def rrf(dense, lexical, k=4):
    scores, docs = {}, {}
    for ranking in (dense, lexical):
        for rank, doc in enumerate(ranking, 1):
            key = (doc.metadata["page"], doc.page_content[:120])
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)[:k]]


def metrics(docs, gold_pages):
    pages = [doc.metadata["page"] for doc in docs]
    first_rank = next((rank for rank, page in enumerate(pages, 1) if page in gold_pages), None)
    return int(first_rank is not None), (1 / first_rank if first_rank else 0), pages


def main():
    if not API_KEY:
        raise EnvironmentError("Set DASHSCOPE_API_KEY first.")
    if not (VECTOR_DIR / "fixed").exists():
        raise FileNotFoundError("Run python ingest.py first.")
    RESULTS_DIR.mkdir(exist_ok=True)
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=API_KEY, base_url=BASE_URL,
                           dimensions=1024, check_embedding_ctx_length=False, chunk_size=10)
    fixed = FAISS.load_local(str(VECTOR_DIR / "fixed"), emb, allow_dangerous_deserialization=True)
    recursive = FAISS.load_local(str(VECTOR_DIR / "recursive"), emb, allow_dangerous_deserialization=True)
    with open(VECTOR_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    methods = {
        "固定分块 + 向量": lambda q: fixed.similarity_search(q, k=4),
        "递归分块 + 向量": lambda q: recursive.similarity_search(q, k=4),
        "递归分块 + RRF混合": lambda q: rrf(recursive.similarity_search(q, k=4), bm25_docs(q, bm25, 4)),
    }
    details, summary = [], []
    for name, search in methods.items():
        recalls, reciprocal_ranks, latencies = [], [], []
        for item in TESTS:
            started = time.perf_counter()
            docs = search(item["question"])
            latency = (time.perf_counter() - started) * 1000
            recall, mrr, pages = metrics(docs, item["gold_pages"])
            recalls.append(recall); reciprocal_ranks.append(mrr); latencies.append(latency)
            details.append({"方案": name, "问题": item["question"], "标注相关页": ",".join(map(str, sorted(item["gold_pages"]))),
                            "检索页": ",".join(map(str, pages)), "Recall@4": recall, "RR@4": round(mrr, 3), "耗时_ms": round(latency, 1)})
        summary.append({"方案": name, "Recall@4": round(sum(recalls) / len(recalls), 3),
                        "MRR@4": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
                        "平均检索耗时_ms": round(sum(latencies) / len(latencies), 1)})

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    write_csv(RESULTS_DIR / "comparison.csv", summary)
    write_csv(RESULTS_DIR / "query_details.csv", details)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    names = [row["方案"] for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, [row["Recall@4"] for row in summary], label="Recall@4", color="#556ee6")
    axes[0].bar(names, [row["MRR@4"] for row in summary], label="MRR@4", color="#33b5e5", alpha=.7)
    axes[0].set_ylim(0, 1.05); axes[0].set_title("检索质量（人工页码标注）"); axes[0].legend()
    axes[1].bar(names, [row["平均检索耗时_ms"] for row in summary], color="#f5a623")
    axes[1].set_title("平均检索耗时"); axes[1].set_ylabel("毫秒")
    for ax in axes: ax.tick_params(axis="x", rotation=12)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "comparison.png", dpi=180)
    print("\n".join(str(row) for row in summary))


if __name__ == "__main__":
    main()
