"""
RAG 问答 HTTP 服务（FastAPI）

提供两类接口：
  /query        — 标准问答，返回答案 + 引用
  /query/debug  — 调试接口，逐步返回每个检索阶段的中间结果
  /             — 可视化 Web 页面

启动：
  cd src
  uvicorn serve:app --host 0.0.0.0 --port 8000

依赖：
  pip install fastapi uvicorn
"""

import json
import importlib.util
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

_PIPELINE_PATH = Path(__file__).parent / "rag_pipeline.py"
_STATIC_DIR    = Path(__file__).parent / "static"
_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"

# 内置公司列表（当 manifest.json 尚不存在时作为兜底）
_FALLBACK_STOCKS = [
    {"stock_code": "600519", "plate": "sh", "company_name": "贵州茅台"},
    {"stock_code": "000858", "plate": "sz", "company_name": "五粮液"},
    {"stock_code": "600887", "plate": "sh", "company_name": "伊利股份"},
    {"stock_code": "601318", "plate": "sh", "company_name": "中国平安"},
    {"stock_code": "600036", "plate": "sh", "company_name": "招商银行"},
    {"stock_code": "300750", "plate": "sz", "company_name": "宁德时代"},
    {"stock_code": "002594", "plate": "sz", "company_name": "比亚迪"},
    {"stock_code": "601012", "plate": "sh", "company_name": "隆基绿能"},
    {"stock_code": "002415", "plate": "sz", "company_name": "海康威视"},
    {"stock_code": "000725", "plate": "sz", "company_name": "京东方A"},
    {"stock_code": "000063", "plate": "sz", "company_name": "中兴通讯"},
    {"stock_code": "000333", "plate": "sz", "company_name": "美的集团"},
    {"stock_code": "000651", "plate": "sz", "company_name": "格力电器"},
    {"stock_code": "600690", "plate": "sh", "company_name": "海尔智家"},
    {"stock_code": "600276", "plate": "sh", "company_name": "恒瑞医药"},
    {"stock_code": "300760", "plate": "sz", "company_name": "迈瑞医疗"},
    {"stock_code": "601888", "plate": "sh", "company_name": "中国中免"},
    {"stock_code": "002714", "plate": "sz", "company_name": "牧原股份"},
    {"stock_code": "000002", "plate": "sz", "company_name": "万科A"},
    {"stock_code": "601857", "plate": "sh", "company_name": "中国石油"},
]

_FALLBACK_YEARS = ["2021", "2022", "2023", "2024", "2025"]


# ── 动态加载 rag_pipeline（importlib 方式，保持与启动目录无关）────────────────

def _load_pipeline_module():
    spec   = importlib.util.spec_from_file_location("rag_pipeline", _PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_module  = None   # 持有模块引用，供 debug 端点访问模块级函数
pipeline = None   # RAGPipeline 实例


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _module, pipeline
    logger.info("服务启动，初始化 RAG Pipeline...")
    _module  = _load_pipeline_module()
    pipeline = _module.RAGPipeline(
        use_bm25          = True,
        use_rerank        = False,
        use_query_rewrite = False,
    )
    logger.info("Pipeline 初始化完成，开始接受请求")
    yield
    logger.info("服务关闭")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "年报 RAG 问答服务",
    description = "FAISS + BM25 混合检索 + DashScope qwen3.7-plus，含调试接口",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str           = Field(...,  example="贵州茅台2023年营业收入是多少")
    stock_code: Optional[str] = Field(None, example="600519")
    year:       Optional[str] = Field(None, example="2023")

class Citation(BaseModel):
    index:    int
    source:   str
    chunk_id: str

class QueryResponse(BaseModel):
    answer:    str
    citations: list[Citation]

# ── Debug 专用模型 ────────────────────────────────────────────────────────────

class VecChunk(BaseModel):
    rank:            int
    vec_score:       float
    source:          str
    content_preview: str

class BM25Chunk(BaseModel):
    rank:            int
    bm25_score:      float
    source:          str
    content_preview: str

class RRFChunk(BaseModel):
    rank:            int
    rrf_score:       float
    vec_rank:        Optional[int]
    bm25_rank:       Optional[int]
    source:          str
    content_preview: str

class ContextChunk(BaseModel):
    index:   int
    source:  str
    content: str          # 完整内容，用于展示"输入 LLM 的原文"

class DebugResponse(BaseModel):
    question:       str
    vec_results:    list[VecChunk]
    bm25_results:   list[BM25Chunk]
    rrf_results:    list[RRFChunk]
    context_chunks: list[ContextChunk]
    answer:         str
    citations:      list[Citation]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _build_source(item: dict) -> str:
    """把 chunk metadata 格式化为可读来源字符串。"""
    s = f"{item.get('stock_code','')} {item.get('year','')}年报"
    section = item.get("section", "")
    if section:
        # 只保留最后两级，避免过长
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num", -1)
    if page and page != -1:
        s += f" · 第{page}页"
    return s

def _preview(text: str, n: int = 150) -> str:
    text = text.strip()
    return text[:n] + "…" if len(text) > n else text

def _filter_meta(req: QueryRequest) -> Optional[dict]:
    fm = {}
    if req.stock_code: fm["stock_code"] = req.stock_code
    if req.year:       fm["year"]        = req.year
    return fm or None


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    """返回可视化页面。"""
    return FileResponse(
        _STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health", summary="健康检查")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return {"status": "ok", "pipeline_ready": True}


@app.get("/companies", summary="获取已下载年报的公司和年份列表")
def companies():
    """返回 manifest 中的公司列表，前端用于填充筛选下拉框。"""
    if _MANIFEST_PATH.exists():
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.loads(f.read())
            stocks = {}
            years = set()
            for item in manifest:
                code = item["stock_code"]
                if code not in stocks:
                    stocks[code] = {
                        "stock_code": code,
                        "company_name": item.get("company_name", code),
                        "plate": item.get("plate", ""),
                    }
                years.add(item.get("year", ""))
            return {
                "stocks": sorted(stocks.values(), key=lambda x: x["stock_code"]),
                "years":  sorted(years, reverse=True),
            }
        except Exception as e:
            logger.warning(f"读取 manifest 失败: {e}")
    return {"stocks": _FALLBACK_STOCKS, "years": _FALLBACK_YEARS}


@app.post("/query", response_model=QueryResponse, summary="标准问答")
def query(req: QueryRequest):
    """标准问答接口，返回最终答案和来源引用。"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    try:
        result = pipeline.query(req.question, filter_meta=_filter_meta(req), verbose=True)
    except Exception as e:
        logger.error(f"Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer    = result["answer"],
        citations = [Citation(**c) for c in result["citations"]],
    )


@app.post("/query/debug", response_model=DebugResponse, summary="调试：逐步返回中间结果")
def query_debug(req: QueryRequest):
    """
    调试接口，逐步执行 RAG 流水线并返回每步的中间结果：

    1. 向量检索（vec_results）：FAISS 召回 Top-10，含余弦相似度分数
    2. BM25 检索（bm25_results）：关键词召回 Top-10，含 BM25 分数
    3. RRF 融合（rrf_results）：两路合并后 Top-5，标注每条来自哪一路
    4. 输入 LLM 的上下文（context_chunks）：最终 Top-4 的完整文本
    5. 生成答案（answer + citations）
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    TOP_K       = _module.TOP_K_RETRIEVE   # 10
    TOP_K_FINAL = _module.TOP_K_RERANK     # 4
    fm          = _filter_meta(req)

    try:
        # ① 向量检索
        vec_results  = pipeline.vec_store.search(req.question, TOP_K, fm)
        vec_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(vec_results, 1)}

        # ② BM25 检索（不受 filter_meta 限制，搜全库后融合）
        bm25_results  = pipeline.bm25_store.search(req.question, TOP_K) if pipeline.bm25_store else []
        bm25_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(bm25_results, 1)}

        # ③ RRF 融合
        if bm25_results:
            candidates = _module.reciprocal_rank_fusion(vec_results, bm25_results)
        else:
            candidates = vec_results

        # ④ 取 Top-K 组装上下文
        final         = candidates[:TOP_K_FINAL]
        context, cits = _module.build_context(final)

        # ⑤ LLM 生成
        answer = _module.call_llm(req.question, context, pipeline.client)

    except Exception as e:
        logger.error(f"Debug Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return DebugResponse(
        question = req.question,
        vec_results = [
            VecChunk(
                rank=i+1,
                vec_score=item.get("vec_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(vec_results[:5])
        ],
        bm25_results = [
            BM25Chunk(
                rank=i+1,
                bm25_score=item.get("bm25_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(bm25_results[:5])
        ],
        rrf_results = [
            RRFChunk(
                rank=i+1,
                rrf_score=item.get("rrf_score", 0.0),
                vec_rank=vec_rank_map.get(item["chunk_id"]),
                bm25_rank=bm25_rank_map.get(item["chunk_id"]),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(candidates[:5])
        ],
        context_chunks = [
            ContextChunk(
                index=i+1,
                source=_build_source(item),
                content=item.get("parent_content") or item["content"],
            ) for i, item in enumerate(final)
        ],
        answer    = answer,
        citations = [Citation(**c) for c in cits],
    )
