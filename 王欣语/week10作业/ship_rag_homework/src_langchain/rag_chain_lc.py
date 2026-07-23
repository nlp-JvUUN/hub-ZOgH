"""
RAG 问答链（LangChain LCEL 版 - 船舶术语）

核心对比点（与原生版 src/rag_pipeline.py）：
┌──────────────────┬──────────────────────┬──────────────────────┐
│ 环节             │ 原生版               │ LangChain 版         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 检索             │ 向量 + BM25 混合     │ FAISS 单路           │
│ Embedding        │ DashScope API        │ 本地 BGE             │
│ 排序             │ RRF + CrossEncoder   │ 相似度得分直接排序   │
│ 链路组织         │ 手写流程控制         │ LCEL pipe (|) 操作符 │
│ 代码量           │ ~300 行              │ ~120 行              │
│ 可调试性         │ 高（每步都可打印）   │ 中（需用 callbacks） │
│ 网络依赖         │ 需要联网             │ 完全离线             │
└──────────────────┴──────────────────────┴──────────────────────┘

依赖：
  pip install langchain langchain-openai langchain-community langchain-huggingface faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"  # 仅 LLM 需要
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODEL_PATH      = "/Users/wangxinyu/Desktop/python/最新/pretrain_models/bge-small-zh-v1.5"

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的船舶/验船术语助手，专门回答关于船舶检验、建造、法规术语的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体术语时，在句末标注来源编号，如：Cathodic Protection 对应阴极防护[1]
4. 涉及中英术语对照时，同时给出中文和英文
5. 回答简洁，重点突出，避免无关废话
6. 对于定义类问题，先给出术语名称，再解释定义"""


# ── 组件初始化 ────────────────────────────────────────────────────────────────

def get_llm():
    """LangChain ChatOpenAI 指向 DashScope。"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=DASHSCOPE_URL,
        temperature=0.1,
    )


def get_embeddings():
    """加载本地 BGE 模型。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"本地模型不存在: {MODEL_PATH}")

    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    """加载已构建的 FAISS 向量库。"""
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在: {VECTORSTORE_DIR}\n"
            "请先运行: python src_langchain/build_index_lc.py"
        )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── LCEL 链构建 ───────────────────────────────────────────────────────────────

def build_chain(vectorstore):
    """构建标准 RAG 链。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            label = f"[{i}]"
            if meta.get("doc_source"):
                label += f" {meta['doc_source']}"
            if meta.get("category"):
                label += f"（{meta['category']}）"
            if meta.get("row_num") and meta["row_num"] != -1:
                label += f" 第{meta['row_num']}行"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，并在引用术语时标注来源编号（如[1]）。"),
    ])

    # LCEL 核心：用 | 串联各组件
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="船舶术语 RAG 问答（LangChain LCEL 版）")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    logger.info("加载本地 BGE embedding 模型...")
    embeddings = get_embeddings()

    logger.info("加载 FAISS 向量库...")
    vectorstore = get_vectorstore(embeddings)

    logger.info("构建 RAG 链...")
    chain = build_chain(vectorstore)

    def run_query(question: str):
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")
        answer = chain.invoke(question)
        print(f"\n{answer}")

    if args.query:
        run_query(args.query)
    else:
        print(f"船舶术语 RAG 问答系统（LangChain LCEL 版）")
        print(f"LLM: {LLM_MODEL}  |  Embedding: 本地 BGE  |  向量库: {VECTORSTORE_DIR}")
        print("特点：完全离线运行（仅 LLM 生成需要联网）")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
