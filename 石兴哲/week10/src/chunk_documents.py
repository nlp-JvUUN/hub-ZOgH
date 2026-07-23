"""
文档分块脚本：对解析后的专利做分块处理

与 week10_rag 的对比差异：
  - 新增 block_type: "abstract"（摘要块，单独处理）
  - 新增 block_type: "claim"（权利要求块，逐条独立保留）
  - 元数据字段: patent_id / assignee / patent_office 替代 stock_code / year
  - 三种分块策略逻辑不变，验证了架构的通用性

三种策略：
  策略A  fixed        — 固定大小 500 字符
  策略B  semantic     — 语义分块（默认）
  策略C  hierarchical — 父子块（Small-to-Big）
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "semantic"   # "fixed" | "semantic" | "hierarchical"


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    """
    按专利结构分块：

    - title:    标题独立成块
    - abstract: 摘要独立成块（高质量的检索入口）
    - claim:    每条权利要求独立成块（法律主张不能切断）
    - text:     段落合并到 max_chunk_size
    """
    buffer_blocks = []
    buffer_len    = 0

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "section":    " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        # 标题/摘要/权利要求 → 单独成块
        if btype in ("title", "abstract", "claim"):
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "section":     " > ".join(block["section_path"]),
                    "block_types": [btype],
                    "claim_num":   block.get("claim_num", 0),
                }
            }
            continue

        # 文字块（说明书段落）→ 累积
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


# ── 策略 C：层级分块（父子块）── 与 week10 完全一致 ────────────────────────────

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size:  int = 400,
    overlap:     int = 50,
) -> Iterator[dict]:
    full_text = "\n\n".join(b["content"] for b in blocks if b["content"].strip())

    parents = []
    start   = 0
    while start < len(full_text):
        end       = min(start + parent_size, len(full_text))
        content   = full_text[start:end]
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
            c_end   = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content":  child_content,
                "metadata": {
                    "parent_id":      p_id,
                    "parent_content": p_content,
                    "block_types":    ["text"],
                    "section":        "",
                }
            }
            c_start += child_size - overlap


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_chunk_id(patent_id: str, idx: int) -> str:
    safe = patent_id.replace("/", "_").replace("\\", "_")
    return f"{safe}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])

    patent_id = meta.get("patent_id", parsed_path.stem)

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    # 根据策略生成 chunks
    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {"block_types": ["text"], "section": ""}
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 补充公共元信息
    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(patent_id, idx)
        chunk["chunk_id"] = chunk_id
        chunk["metadata"]["patent_id"]     = patent_id
        chunk["metadata"]["title"]         = meta.get("title", "")
        chunk["metadata"]["assignee"]      = meta.get("assignee", "")
        chunk["metadata"]["patent_office"] = meta.get("patent_office", "")
        chunk["metadata"]["strategy"]      = strategy
        chunk["metadata"]["source_file"]   = parsed_path.name
        result.append(chunk)

    # 保存
    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result


def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_patents.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    # 合并
    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    # 统计
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    claim_count = sum(1 for c in all_chunks if "claim" in c["metadata"].get("block_types", []))
    abstract_count = sum(1 for c in all_chunks if "abstract" in c["metadata"].get("block_types", []))
    logger.info(f"其中: 权利要求块={claim_count}  摘要块={abstract_count}")


if __name__ == "__main__":
    main()
