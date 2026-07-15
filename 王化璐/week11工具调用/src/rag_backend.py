"""
rag_backend.py — AI技术面试问答 RAG 检索后端（三种方式共享的业务逻辑）

核心设计：
  1. 纯业务逻辑层，不感知被哪种方式调用（Function Call / MCP / CLI 都复用它）
  2. 模块级加载 FAISS 索引一次，进程内所有调用复用（CLI 子进程每次启动加载一次）
  3. L2 归一化 + IndexFlatIP 内积 = 余弦相似度
  4. 元数据过滤（title / topic）在检索后做，过滤条件越多搜回数越少

使用方式（作为模块）：
  from src.rag_backend import search_ai_knowledge, list_papers
  print(search_ai_knowledge("Transformer自注意力机制原理", topic="Transformer架构", top_k=3))

依赖：
  pip install faiss-cpu numpy openai
  向量数据位于 vectorstore/（运行 scripts/copy_data.py 复制）
  环境变量：DASHSCOPE_API_KEY（Embedding 用）

知识库说明：
  论文（title）：Attention Is All You Need / BERT / GPT-3 / InstructGPT ChatGPT / LLaMA / RAG / 动手学深度学习
  主题（topic）：Transformer架构 / 预训练语言模型 / 大语言模型 / 指令微调 / 开源大模型 / 检索增强生成 / 深度学习教程
  规模：7 份文档，共多个语义分块
"""

import json
import os
import sys
from pathlib import Path

# Windows 上 torch 与 numpy 各自链接 OpenMP 会冲突，必须打开此开关
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from openai import OpenAI

# ── 常量 ──────────────────────────────────────────────────────────────────

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024

# 用 __file__ 定位项目根目录，无论从哪个工作目录启动都能找到 vectorstore/
BASE_DIR = Path(__file__).parent.parent

_local_index = Path("vectorstore") / "faiss_index.bin"
_local_meta = Path("vectorstore") / "faiss_meta.json"

if _local_index.exists() and _local_meta.exists():
    FAISS_INDEX_PATH = str(_local_index)
    FAISS_META_PATH = str(_local_meta)
else:
    FAISS_INDEX_PATH = str(BASE_DIR / "vectorstore" / "faiss_index.bin")
    FAISS_META_PATH = str(BASE_DIR / "vectorstore" / "faiss_meta.json")

# 论文信息表（用于 list_papers 和参数说明）
PAPERS = [
    {"title": "Attention Is All You Need", "topic": "Transformer架构", "year": "2017"},
    {"title": "BERT", "topic": "预训练语言模型", "year": "2018"},
    {"title": "GPT-3", "topic": "大语言模型", "year": "2020"},
    {"title": "InstructGPT ChatGPT", "topic": "指令微调", "year": "2022"},
    {"title": "LLaMA", "topic": "开源大模型", "year": "2023"},
    {"title": "RAG (检索增强生成) 原始论文", "topic": "检索增强生成", "year": "2020"},
    {"title": "动手学深度学习 (PyTorch版)", "topic": "深度学习教程", "year": "2023"},
]

# ── 初始化（模块导入时执行一次）────────────────────────────────────────────

if not DASHSCOPE_API_KEY:
    print("错误：未设置环境变量 DASHSCOPE_API_KEY", file=sys.stderr)
    sys.exit(1)

_embed_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

try:
    import faiss
except ImportError:
    print("错误：未安装 faiss-cpu，请运行 pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

if not Path(FAISS_INDEX_PATH).exists() or not Path(FAISS_META_PATH).exists():
    print(f"错误：向量索引文件不存在，请先运行 scripts/copy_data.py", file=sys.stderr)
    print(f"  期望路径：{FAISS_INDEX_PATH}", file=sys.stderr)
    sys.exit(1)

_index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    _meta_list: list[dict] = json.load(f)

print(
    f"[rag_backend] 就绪：{_index.ntotal} 个向量，{len(_meta_list)} 条元数据",
    file=sys.stderr,
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """
    调用 DashScope Embedding API，返回 L2 归一化后的 float32 向量。
    FAISS 使用 IndexFlatIP（内积），预先 L2 归一化后内积等价于余弦相似度。
    """
    response = _embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── 对外接口 ──────────────────────────────────────────────────────────────

def search_ai_knowledge(
    query: str,
    title: str | None = None,
    topic: str | None = None,
    top_k: int = 5,
) -> str:
    """
    在AI技术面试知识库中检索与问题最相关的段落。

    Args:
        query:   检索问题，自然语言，例如 "Transformer自注意力机制原理"
        title:   可选，按论文标题过滤。如 "Attention Is All You Need" / "BERT" / "GPT-3"
        topic:   可选，按主题过滤。如 "Transformer架构" / "预训练语言模型" / "大语言模型"
        top_k:   返回段落数，默认5，建议不超过10

    Returns:
        按相关度排序的段落列表，每段含来源（论文标题、年份、章节、页码）
    """
    try:
        query_vec = get_embedding(query)
    except Exception as e:
        return f"Embedding 调用失败：{e}"

    # 有过滤条件时多搜几倍，再过滤；无过滤时搜略多一点
    search_k = min(top_k * 10 if (title or topic) else top_k * 3, _index.ntotal)
    distances, indices = _index.search(query_vec.reshape(1, -1), search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_meta_list):
            continue
        meta = _meta_list[idx]
        if title and meta.get("title") != title:
            continue
        if topic and meta.get("topic") != topic:
            continue
        results.append({
            "score": float(dist),
            "content": meta.get("content", ""),
            "title": meta.get("title", ""),
            "year": str(meta.get("year", "")),
            "section": meta.get("section", ""),
            "page_num": meta.get("page_num", ""),
        })
        if len(results) >= top_k:
            break

    if not results:
        filter_parts = []
        if title:
            filter_parts.append(f"论文={title}")
        if topic:
            filter_parts.append(f"主题={topic}")
        filter_str = f"（过滤条件：{', '.join(filter_parts)}）" if filter_parts else ""
        return f"未找到相关内容{filter_str}，请尝试换一种问法或去掉过滤条件"

    lines = [f"检索到 {len(results)} 条相关段落：\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"【{i}】{r['title']}（{r['year']}）"
            f" | 第{r['page_num']}页 | 相关度：{r['score']:.3f}"
        )
        lines.append(f"章节：{r['section']}")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines)


def list_papers() -> str:
    """
    列出AI技术面试知识库中包含的所有论文及主题。

    Returns:
        论文列表，含标题、主题、年份
    """
    lines = ["AI技术面试知识库收录论文列表：\n"]
    for p in PAPERS:
        lines.append(f"  {p['title']}  | 主题：{p['topic']}  | 年份：{p['year']}")
    lines.append(f"\n共 {len(PAPERS)} 份文档")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：直接运行看检索结果
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("search")
    p1.add_argument("--query", required=True)
    p1.add_argument("--title", default=None)
    p1.add_argument("--topic", default=None)
    p1.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list-papers")
    args = parser.parse_args()

    if args.cmd == "search":
        print(search_ai_knowledge(args.query, args.title, args.topic, args.top_k))
    else:
        print(list_papers())