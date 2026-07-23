"""
rag_server.py — AI技术面试问答 RAG 检索 MCP Server（方式二：MCP）

核心设计：
  1. MCP Server 把"现成业务逻辑"封装成协议工具：函数体直接复用 src/rag_backend
     ——零逻辑重复，只加一层 @mcp.tool() 协议装饰
  2. Python 函数签名（类型注解 + docstring）自动生成 JSON Schema 供 LLM 决策
  3. 所有 print/log 必须写 stderr：stdout 是 MCP JSON-RPC 通道，混入普通文本会破坏连接

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/rag_server.py

依赖：
  pip install mcp faiss-cpu numpy openai
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""

import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（子进程 cwd 不一定是项目根）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 注意：用 as 别名导入后端函数，避免下方同名 tool 函数遮蔽后递归调用自己
from src.rag_backend import (  # noqa: E402
    search_ai_knowledge as _search_ai_knowledge,
    list_papers as _list_papers,
)


def log(msg: str):
    # stdout 是协议通道，所有日志必须写 stderr
    print(msg, file=sys.stderr, flush=True)


# FastMCP 实例，name 是这个 Server 的身份标识，Client 连接后会收到
mcp = FastMCP("rag-server")


@mcp.tool()
def search_ai_knowledge(
    query: str,
    title: str | None = None,
    topic: str | None = None,
    top_k: int = 5,
) -> str:
    """
    在AI技术面试知识库中检索与问题最相关的段落。

    知识库仅收录以下论文：
    Attention Is All You Need(Transformer架构)/BERT(预训练语言模型)/GPT-3(大语言模型)/
    InstructGPT ChatGPT(指令微调)/LLaMA(开源大模型)/RAG(检索增强生成)/动手学深度学习(深度学习教程)
    不在库内的主题请勿调用本工具。

    Args:
        query:   检索问题。重要：不要包含论文标题和主题（已由 title/topic 过滤），
                 只用简短技术术语，例如 '自注意力机制'、'预训练目标'、'少样本学习'。
                 把标题写进 query 会稀释检索精度。
        title:   可选，按论文标题过滤，如 'Attention Is All You Need'。
        topic:   可选，按主题过滤：'Transformer架构' / '预训练语言模型' / '大语言模型'
                 / '指令微调' / '开源大模型' / '检索增强生成' / '深度学习教程'。
        top_k:   返回段落数，默认5，建议不超过10。

    Returns:
        按相关度排序的段落列表，每段含来源（论文标题、年份、章节、页码）。
    """
    return _search_ai_knowledge(query, title, topic, top_k)


@mcp.tool()
def list_papers() -> str:
    """
    列出AI技术面试知识库中收录的所有论文、主题与年份。
    用于确认目标论文在库内，并获取正确的 title/topic 参数。

    Returns:
        论文列表，含标题、主题、年份。
    """
    return _list_papers()


if __name__ == "__main__":
    log("RAG MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
