"""
FastAPI service for the single-book QA system.

Start:
  python -m uvicorn src.book_qa_serve:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.book_qa_pipeline import BookQAPipeline, CHUNKS_PATH, INDEX_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_PATH = Path(__file__).parent / "static" / "book_qa.html"

pipeline: Optional[BookQAPipeline] = None
startup_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, startup_error
    try:
        pipeline = BookQAPipeline()
        startup_error = None
    except Exception as e:
        pipeline = None
        startup_error = str(e)
        logger.warning("书籍问答索引未就绪: %s", e)
    yield


app = FastAPI(
    title="《性能之巅》问答系统",
    description="基于单本 PDF 的本地向量检索 + 可选 LLM 生成问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., example="什么是 USE 方法？")


class Citation(BaseModel):
    index: int
    source: str
    chunk_id: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok" if pipeline else "index_not_ready",
        "pipeline_ready": pipeline is not None,
        "index_path": str(INDEX_PATH),
        "chunks_path": str(CHUNKS_PATH),
        "error": startup_error,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                startup_error
                or "索引尚未构建，请先运行 python src/book_qa_pipeline.py build"
            ),
        )

    try:
        result = pipeline.answer(req.question)
    except Exception as e:
        logger.error("问答失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        mode=result["mode"],
    )
