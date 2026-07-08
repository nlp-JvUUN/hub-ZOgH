"""
文档分块脚本：兼容PDF/docx/md/txt通用解析结果
三种分块策略：固定大小、语义分块、层级父子块
输出chunk结构与原逻辑完全一致，向量库/RAG无需改动
"""
import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── 策略A：固定大小分块 ──────────────────────────────────────────────────────
def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap

# ── 策略B：语义分块 ──────────────────────────────────────────────────────────
def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    buffer_blocks = []
    buffer_len    = 0
    buffer_meta   = {}

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "page_num":   buf[0].get("page_num", 0),
            "section":    " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr":     buf[0].get("is_ocr", False),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])
        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block.get("page_num", 0),
                    "section":  " > ".join(block["section_path"]),
                    "block_types": ["table"],
                    "is_ocr":   block.get("is_ocr", False),
                }
            }
            continue
        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0
        buffer_blocks.append(block)
        buffer_len += blen
    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result

# ── 策略C：层级分块（父子块 Small-to-Big） ───────────────────────────────────
def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size:  int = 400,
    overlap:     int = 50,
) -> Iterator[dict]:
    full_text  = "\n\n".join(b["content"] for b in blocks if b["content"].strip())
    parents = []
    start   = 0
    while start < len(full_text):
        end     = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content":   content,
            "start":     start,
            "end":       end,
        })
        start += parent_size - overlap
    for parent in parents:
        p_content = parent["content"]
        p_id      = parent["parent_id"]
        c_start   = 0
        while c_start < len(p_content):
            c_end     = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content":  child_content,
                "metadata": {
                    "parent_id":   p_id,
                    "parent_content": p_content,
                    "block_types": ["text"],
                    "is_ocr":      False,
                    "section":     "",
                    "page_num":    0,
                }
            }
            c_start += child_size - overlap

# ── 主流程配置 ────────────────────────────────────────────────────────────────
STRATEGY = "semantic"

def build_chunk_id(file_name: str, idx: int) -> str:
    """通用chunk_id，不再依赖股票代码，兼容任意文件"""
    clean_name = "".join(c for c in file_name if c not in r'\/:*?"<>|')
    return f"{clean_name}_{idx:05d}"

def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)
    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])
    filename = meta.get("filename", "unknown_file")
    stock_code = meta.get("stock_code", "unknown")
    year       = meta.get("year", "unknown")
    file_suffix = meta.get("file_suffix", "")
    logger.info(f"分块 {parsed_path.name}  文件类型={file_suffix} 策略={strategy} blocks={len(blocks)}")
    raw_chunks = []
    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {"block_types": ["text"], "is_ocr": False, "section": "", "page_num": 0}
            })
    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)
    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)
    else:
        raise ValueError(f"未知策略: {strategy}")
    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(filename, idx)
        chunk["chunk_id"]              = chunk_id
        chunk["metadata"]["stock_code"] = stock_code
        chunk["metadata"]["year"]       = year
        chunk["metadata"]["strategy"]   = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        chunk["metadata"]["file_suffix"] = file_suffix
        result.append(chunk)
    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result

def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_file.py")
        return
    all_chunks = []
    pdf_cnt = 0
    docx_cnt = 0
    txtmd_cnt = 0
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)
        # 统计文件类型
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
            suf = d.get("meta", {}).get("file_suffix", "")
            if suf == ".pdf":
                pdf_cnt += 1
            elif suf == ".docx":
                docx_cnt += 1
            else:
                txtmd_cnt += 1
    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")
    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    ocr_count   = sum(1 for c in all_chunks if c["metadata"].get("is_ocr"))
    logger.info(f"文件统计：PDF={pdf_cnt} Word={docx_cnt} TXT/MD={txtmd_cnt}")
    logger.info(f"其中表格块: {table_count}  OCR块: {ocr_count}")

if __name__ == "__main__":
    main()