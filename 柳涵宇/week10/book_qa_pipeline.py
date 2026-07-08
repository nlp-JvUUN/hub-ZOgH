"""
Single-book RAG pipeline for ``性能之巅.pdf``.

Features:
  - Extract text from a PDF, with optional OCR for scanned pages.
  - Chunk pages into retrieval units.
  - Build a local FAISS index using the bundled bge-small-zh-v1.5 model.
  - Answer questions with OpenAI/DashScope when a key is configured, or return
    extractive evidence when running fully offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DEFAULT_PDF_PATH = Path(os.getenv("BOOK_QA_PDF_PATH", r"D:\BaiduNetdiskDownload\性能之巅.pdf"))
INDEX_DIR = BASE_DIR / "vectorstore" / "performance_book"
PARSED_PATH = INDEX_DIR / "pages.json"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
INDEX_PATH = INDEX_DIR / "index.faiss"
MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"
LOCAL_TESSDATA_DIR = BASE_DIR / "tools" / "tessdata"
DEFAULT_TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

CHUNK_SIZE = int(os.getenv("BOOK_QA_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("BOOK_QA_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("BOOK_QA_TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("BOOK_QA_SCORE_THRESHOLD", "0.22"))

SYSTEM_PROMPT = """你是《性能之巅》这本书的阅读问答助手。

回答规则：
1. 只根据【参考资料】回答，不要编造书中没有的内容。
2. 如果资料不足，直接说明“根据当前检索到的内容无法回答”。
3. 回答要清晰、结构化，涉及具体说法时标注来源编号，如[1]。
4. 可以用自己的话总结，但不要脱离参考资料。"""


@dataclass
class Chunk:
    chunk_id: str
    content: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_faiss_read(path: Path):
    import faiss

    try:
        return faiss.read_index(str(path))
    except RuntimeError as e:
        if all(ord(ch) < 128 for ch in str(path)):
            raise
        tmp_path = Path(tempfile.gettempdir()) / "performance_book_index.faiss"
        shutil.copyfile(path, tmp_path)
        logger.warning("FAISS 读取中文路径失败，已复制到临时路径后加载: %s", e)
        return faiss.read_index(str(tmp_path))


def _load_embedder():
    from sentence_transformers import SentenceTransformer

    model_name = str(MODEL_PATH) if MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return SentenceTransformer(model_name)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = _load_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def _find_tesseract_cmd() -> Optional[str]:
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd
    if DEFAULT_TESSERACT_EXE.exists():
        return str(DEFAULT_TESSERACT_EXE)
    return None


def _tessdata_config() -> str:
    tessdata_dir = Path(os.getenv("TESSDATA_PREFIX", str(LOCAL_TESSDATA_DIR)))
    if tessdata_dir.exists() and any(ord(ch) >= 128 for ch in str(tessdata_dir)):
        tmp_tessdata = Path(tempfile.gettempdir()) / "rag_book_tessdata"
        tmp_tessdata.mkdir(parents=True, exist_ok=True)
        for name in ("chi_sim.traineddata", "eng.traineddata", "osd.traineddata"):
            src = tessdata_dir / name
            dst = tmp_tessdata / name
            if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
                shutil.copyfile(src, dst)
        tessdata_dir = tmp_tessdata
    return f"--tessdata-dir {tessdata_dir}" if tessdata_dir.exists() else ""


def _ocr_page(page, lang: str, dpi: int) -> str:
    tesseract_cmd = _find_tesseract_cmd()
    if not tesseract_cmd:
        raise RuntimeError(
            "这份 PDF 是扫描版，但当前系统没有安装 tesseract 可执行程序。\n"
            "请安装 Tesseract OCR，并确认命令行能运行 `tesseract --version`。\n"
            "中文书籍还需要安装 chi_sim 语言包。"
        )
    import pytesseract
    import fitz

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    from PIL import Image

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image, lang=lang, config=_tessdata_config()).strip()


def extract_pages(
    pdf_path: Path = DEFAULT_PDF_PATH,
    parsed_path: Path = PARSED_PATH,
    *,
    force: bool = False,
    max_pages: Optional[int] = None,
    ocr: str = "auto",
    ocr_lang: str = "chi_sim+eng",
    dpi: int = 160,
) -> list[dict]:
    """Extract page text from PDF. OCR is used for image-only pages when enabled."""
    if parsed_path.exists() and not force:
        return _read_json(parsed_path)

    import fitz

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    page_count = min(doc.page_count, max_pages or doc.page_count)
    logger.info("开始解析 PDF: %s，共 %s 页，本次处理 %s 页", pdf_path, doc.page_count, page_count)

    pages: list[dict] = []
    scanned_pages = 0
    for idx in range(page_count):
        page = doc[idx]
        text = page.get_text("text").strip()
        used_ocr = False

        if len(text) < 20:
            scanned_pages += 1
            if ocr in {"auto", "tesseract"}:
                text = _ocr_page(page, ocr_lang, dpi)
                used_ocr = True
            elif ocr == "none":
                text = ""
            else:
                raise ValueError("ocr 参数只能是 auto、tesseract 或 none")

        pages.append({"page": idx + 1, "text": text, "used_ocr": used_ocr})
        if (idx + 1) % 25 == 0:
            logger.info("解析进度: %s/%s 页", idx + 1, page_count)

    if scanned_pages and all(not p["text"] for p in pages):
        raise RuntimeError("PDF 页面未抽取到文本。它看起来是扫描版，请启用并安装 OCR 后重试。")

    _write_json(parsed_path, pages)
    logger.info("页面文本已保存: %s", parsed_path)
    return pages


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = max(text.rfind("。", start, end), text.rfind("\n", start, end))
        if cut > start + size * 0.55:
            end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def build_chunks(pages: list[dict], chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    chunks: list[Chunk] = []
    for page in pages:
        for local_idx, piece in enumerate(_split_text(page["text"]), 1):
            chunks.append(
                Chunk(
                    chunk_id=f"p{page['page']:04d}_{local_idx:02d}",
                    content=piece,
                    page_start=page["page"],
                    page_end=page["page"],
                )
            )

    if not chunks:
        raise RuntimeError("没有生成任何文本块，请检查 PDF 是否已 OCR。")

    data = [c.to_dict() for c in chunks]
    _write_json(chunks_path, data)
    logger.info("已生成 %s 个 chunks: %s", len(data), chunks_path)
    return data


def build_index(chunks: list[dict], index_path: Path = INDEX_PATH) -> None:
    import faiss

    texts = [c["content"] for c in chunks]
    embeddings = encode_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(index_path))
    except RuntimeError as e:
        if all(ord(ch) < 128 for ch in str(index_path)):
            raise
        tmp_path = Path(tempfile.gettempdir()) / "performance_book_index.faiss"
        faiss.write_index(index, str(tmp_path))
        shutil.copyfile(tmp_path, index_path)
        logger.warning("FAISS 无法直接写入中文路径，已先写入临时路径再复制回来: %s", e)
    logger.info("FAISS 索引已保存: %s，共 %s 条，维度 %s", index_path, index.ntotal, dim)


def build_all(args) -> None:
    pages = extract_pages(
        Path(args.pdf),
        PARSED_PATH,
        force=args.force,
        max_pages=args.max_pages,
        ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        dpi=args.dpi,
    )
    chunks = build_chunks(pages)
    build_index(chunks)


class BookQAPipeline:
    def __init__(self):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "书籍索引尚未构建。请先运行: python src/book_qa_pipeline.py build"
            )

        self.embedder = _load_embedder()
        self.index = _safe_faiss_read(INDEX_PATH)
        self.chunks = _read_json(CHUNKS_PATH)
        logger.info("书籍问答索引加载完成，共 %s 个 chunks", len(self.chunks))

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        query_vec = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def answer(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        if not retrieved or retrieved[0]["score"] < SCORE_THRESHOLD:
            return {
                "answer": "根据当前书籍索引未检索到足够相关的内容。",
                "citations": [],
                "retrieved": retrieved,
                "mode": "retrieval",
            }

        context, citations = build_context(retrieved)
        llm_answer = call_llm_if_available(question, context)
        if llm_answer:
            answer = llm_answer
            mode = "llm"
        else:
            answer = build_extractive_answer(retrieved)
            mode = "retrieval"

        return {
            "answer": answer,
            "citations": citations,
            "retrieved": retrieved,
            "mode": mode,
        }


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    citations = []
    for i, item in enumerate(chunks, 1):
        label = f"[{i}] 第{item['page_start']}页"
        parts.append(f"{label}\n{item['content']}")
        citations.append(
            {
                "index": i,
                "source": label,
                "chunk_id": item["chunk_id"],
                "page": item["page_start"],
                "score": item.get("score", 0.0),
            }
        )
    return "\n\n---\n\n".join(parts), citations


def build_extractive_answer(chunks: list[dict]) -> str:
    lines = ["当前未配置大模型 API Key，先返回检索到的书中相关片段："]
    for i, item in enumerate(chunks[:3], 1):
        text = item["content"].replace("\n", " ")
        if len(text) > 320:
            text = text[:320] + "..."
        lines.append(f"\n[{i}] 第{item['page_start']}页：{text}")
    return "\n".join(lines)


def _client_config() -> Optional[tuple[str, str, dict]]:
    from openai import OpenAI

    if os.getenv("OPENAI_API_KEY"):
        return ("openai", os.getenv("BOOK_QA_LLM_MODEL", "gpt-4.1-mini"), {"api_key": os.getenv("OPENAI_API_KEY")})
    if os.getenv("DASHSCOPE_API_KEY") and os.getenv("DASHSCOPE_API_KEY") != "placeholder-key":
        return (
            "dashscope",
            os.getenv("BOOK_QA_LLM_MODEL", "qwen-plus"),
            {
                "api_key": os.getenv("DASHSCOPE_API_KEY"),
                "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            },
        )
    return None


def call_llm_if_available(question: str, context: str) -> Optional[str]:
    config = _client_config()
    if not config:
        return None

    from openai import OpenAI

    provider, model, kwargs = config
    try:
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n请基于参考资料回答。",
                },
            ],
            temperature=0.1,
        )
        logger.info("LLM answer generated via %s/%s", provider, model)
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning("LLM 调用失败，降级为检索片段回答: %s", e)
        return None


def query_once(question: str) -> None:
    result = BookQAPipeline().answer(question)
    print("\n" + result["answer"])
    if result["citations"]:
        print("\n来源：")
        for c in result["citations"]:
            print(f"  {c['source']} · {c['chunk_id']} · score={c['score']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="《性能之巅》单书 RAG 问答")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_parser = sub.add_parser("build", help="抽取文本、分块并构建索引")
    build_parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH))
    build_parser.add_argument("--force", action="store_true", help="重新抽取 PDF")
    build_parser.add_argument("--max-pages", type=int, default=None, help="调试用：只处理前 N 页")
    build_parser.add_argument("--ocr", choices=["auto", "tesseract", "none"], default="auto")
    build_parser.add_argument("--ocr-lang", default="chi_sim+eng")
    build_parser.add_argument("--dpi", type=int, default=160)

    query_parser = sub.add_parser("query", help="命令行提问")
    query_parser.add_argument("question")

    args = parser.parse_args()
    if args.cmd == "build":
        build_all(args)
    elif args.cmd == "query":
        query_once(args.question)


if __name__ == "__main__":
    main()
