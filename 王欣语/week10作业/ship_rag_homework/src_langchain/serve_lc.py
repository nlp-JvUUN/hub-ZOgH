"""
船舶术语 RAG Web 服务（LangChain 版）

与原生版 serve.py 的区别：
  - 使用 LangChain LCEL 链处理请求
  - Embedding 完全本地（BGE），无需 DashScope API Key 即可检索
  - 仅 LLM 生成需要 DashScope API Key

启动：
  uvicorn src_langchain.serve_lc:app --host 0.0.0.0 --port 8001

访问：
  http://localhost:8001
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODEL_PATH      = "/Users/wangxinyu/Desktop/python/最新/pretrain_models/bge-small-zh-v1.5"
STATIC_DIR      = BASE_DIR / "src" / "static"  # 复用原生版的静态页面

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"

# ── 全局状态 ──────────────────────────────────────────────────────────────────

chain = None   # LCEL RAG 链


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    logger.info("服务启动，初始化 LangChain RAG...")

    # 1. 加载本地 Embedding
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("本地 BGE Embedding 加载完成")

    # 2. 加载向量库
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS 向量库加载完成")

    # 3. 构建 LCEL 链
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("未设置 DASHSCOPE_API_KEY，LLM 功能将不可用")
        llm = None
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=api_key,
            openai_api_base=DASHSCOPE_URL,
            temperature=0.1,
        )
        logger.info("DashScope LLM 初始化完成")

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

    SYSTEM_PROMPT = """你是一个专业的船舶/验船术语助手，专门回答关于船舶检验、建造、法规术语的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体术语时，在句末标注来源编号，如：Cathodic Protection 对应阴极防护[1]
4. 涉及中英术语对照时，同时给出中文和英文
5. 回答简洁，重点突出，避免无关废话
6. 对于定义类问题，先给出术语名称，再解释定义"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，并在引用术语时标注来源编号（如[1]）。"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    if llm:
        chain = (
            {
                "context":  retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("LangChain RAG 链构建完成，开始接受请求")
    else:
        chain = None
        logger.warning("LLM 未初始化，仅提供健康检查接口")

    yield
    logger.info("服务关闭")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "船舶术语 RAG 问答服务（LangChain 版）",
    description = "本地 BGE Embedding + DashScope LLM，完全离线检索",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., example="什么是载重线标志")

class QueryResponse(BaseModel):
    answer: str


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    """返回教学可视化页面（复用原生版）。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", summary="健康检查")
def health():
    if chain is None:
        raise HTTPException(status_code=503, detail="Chain 尚未初始化（请检查 DASHSCOPE_API_KEY）")
    return {"status": "ok", "chain_ready": True, "embedding": "本地 BGE", "llm": LLM_MODEL}


@app.post("/query", response_model=QueryResponse, summary="标准问答")
def query(req: QueryRequest):
    """标准问答接口，返回最终答案。"""
    if chain is None:
        raise HTTPException(status_code=503, detail="Chain 尚未初始化")
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        logger.error(f"Chain 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(answer=answer)
