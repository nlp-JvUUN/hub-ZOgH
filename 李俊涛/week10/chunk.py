"""
文档分块脚本：对《CS/大数据/AI/IoT 综合学习指南》PDF 做分块处理


输出格式说明：
  每个 chunk 是一个 dict，包含：
    - chunk_id      唯一标识
    - content       文本内容（供 embedding）
    - metadata      元信息（供过滤/溯源）
      - chapter     所属章节号（如 "第一章"、"2.1"）
      - section     章节路径（字符串，如 "第二章 > 2.3 Apache Spark"）
      - page_num    来源页码
      - strategy    分块策略名
"""

import json
import re
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)




# 本 PDF 的标题编号规律：
#   顶级章：第一章、第二章...第六章（中文数字+章）
#   一级节：1.1、2.3、4.2（阿拉伯数字.阿拉伯数字）
#   二级节：1.1.1、2.2.1、4.2.1（三级编号）
CHAPTER_PATTERN  = re.compile(r"^第[一二三四五六七八九十]+章")       # 第一章
SECTION_PATTERN  = re.compile(r"^\d+\.\d+\s")                       # 1.1 xxx
SUBSEC_PATTERN   = re.compile(r"^\d+\.\d+\.\d+\s")                  # 1.1.1 xxx


def detect_title_level(line: str) -> int | None:
    """
    检测一行文本的标题层级：
      返回 1 = 顶级章（第X章）
      返回 2 = 一级节（X.Y）
      返回 3 = 二级节（X.Y.Z）
      返回 None = 不是标题
    """
    line = line.strip()
    if CHAPTER_PATTERN.match(line):
        return 1
    if SUBSEC_PATTERN.match(line):
        return 3
    if SECTION_PATTERN.match(line):
        return 2
    return None


# ── 策略 A：固定大小分块

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。

    缺点：无视句子/段落边界，可能会切断完整语义。
    优点：实现最简单，块大小可预测。

    参数说明：
      chunk_size : 每个块的目标字符数（500 ≈ 一段中等长度文字）
      overlap    : 相邻块重叠字符数（防止关键词被切断）
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块 

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    """
    按解析结构分块：遇到标题强制切块，段落尽量合并到 max_chunk_size 以内。

    核心思路：
      1. 遇到 "第X章" 或 "X.Y" 标题 → 先把之前积累的内容 flush 成一个 chunk
      2. 普通文字段落累积，超过 max_chunk_size 就 flush

    优点：保留语义完整性，章节边界清晰。
    缺点：块大小不均匀。
    """
    buffer_blocks: list[dict] = []
    buffer_len = 0

    def flush(buf: list[dict]) -> dict | None:
        """把缓冲区里的 blocks 合并成一个 chunk。"""
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        # 元信息取第一个 block 的
        meta = {
            "page_num": buf[0]["page_num"],
            "section":  " > ".join(buf[0]["section_path"]) if buf[0].get("section_path") else "",
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        # ── 标题块：强制先 flush，标题与后续内容分开 ──
        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0

        # ── 文字块：累积，超过 max_chunk_size 就 flush ──
        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0

        buffer_blocks.append(block)
        buffer_len += blen

    # 处理尾部剩余
    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


# ── 策略 C：层级分块（父子块）

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size:  int = 400,
    overlap:     int = 50,
) -> Iterator[dict]:
    """
    两级结构（Small-to-Big Retrieval）：
      父块（parent）：大段落，给 LLM 提供充足上下文
      子块（child） ：小段落，用于向量检索（更精确）

    检索流程：
      用户查询 → 向量检索命中子块 → 取 parent_id → 把父块内容喂给 LLM

    输出的每个 child 块里带 parent_id 和 parent_content 字段。
    """
    # 拼合全文（跳过空块）
    full_text = "\n\n".join(b["content"] for b in blocks if b["content"].strip())

    # ── 第一级：切父块 ──
    parents = []
    start = 0
    while start < len(full_text):
        end = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content":   content,
            "start":     start,
            "end":       end,
        })
        start += parent_size - overlap

    # ── 第二级：从每个父块里切子块 ──
    for parent in parents:
        p_content = parent["content"]
        p_id      = parent["parent_id"]
        c_start   = 0
        while c_start < len(p_content):
            c_end = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content": child_content,
                "metadata": {
                    "parent_id":      p_id,
                    "parent_content": p_content,   # 存全量，供 LLM 读
                    "section":        "",
                    "page_num":       -1,
                }
            }
            c_start += child_size - overlap


# ── 工具函数 

def build_chunk_id(chapter: str, idx: int) -> str:
    """
    生成唯一 chunk ID，使用章节号代替全局序号。
    格式：{章节号}_{章内序号}  例如 "第一章_001"、"第四章_023"
    无章节时降级为 "other_{序号}"。
    """
    label = chapter if chapter else "other"
    return f"{label}_{idx:03d}"


def extract_chapter_from_section(section_path: list[str]) -> str:
    """
    从 section_path 里提取所属章节号。
    例如 ["第一章 计算机基础", "1.1 计算机体系结构"] → "第一章"
    """
    if section_path:
        first = section_path[0]
        m = CHAPTER_PATTERN.match(first)
        if m:
            return m.group(0)
    return ""


# ── 主流程 

STRATEGY = "semantic"   # 改成 "fixed" 或 "hierarchical" 体验不同策略


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    """
    读取一个解析后的 JSON 文件，按指定策略分块，保存到 chunks/ 目录。
    """
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])

    doc_name = parsed_path.stem           # 文件名（不含扩展名）
    doc_title = meta.get("title", doc_name)

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    # ── 根据策略生成 chunks ──
    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {
                    "section":  "",
                    "page_num": -1,
                }
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知策略: {strategy}")

    # ── 补充公共元信息 ──
    result = []
    chapter_idx: dict[str, int] = {}   # 每章内部的计数器

    for chunk in raw_chunks:
        # 先提取所属章节
        section_path = chunk["metadata"].get("section", "")
        chapter = ""
        if section_path:
            parts = section_path.split(" > ")
            if parts and CHAPTER_PATTERN.match(parts[0]):
                chapter = parts[0]

        # 按章节生成 chunk_id：第一章_001, 第一章_002, 第二章_001 ...
        chapter_idx[chapter] = chapter_idx.get(chapter, 0) + 1
        chunk_id = build_chunk_id(chapter, chapter_idx[chapter])

        chunk["chunk_id"]               = chunk_id
        chunk["metadata"]["doc_name"]    = doc_name
        chunk["metadata"]["doc_title"]   = doc_title
        chunk["metadata"]["chapter"]     = chapter
        chunk["metadata"]["strategy"]    = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        result.append(chunk)

    # ── 保存到 chunks/ 目录 ──
    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  -> {len(result)} 个 chunk，已保存 {out_path.name}")
    return result




def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error(
            "没有找到解析结果，请先运行 parse_pdf.py\n"
            f"  期望目录: {PARSED_DIR}"
        )
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    # 合并所有文档的 chunk 到一个文件，方便统一建索引
    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk -> {combined_path}")

    # ── 简单统计 ──
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    # 按章节统计
    chapter_counts: dict[str, int] = {}
    for c in all_chunks:
        ch = c["metadata"].get("chapter", "未分类")
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
    if chapter_counts:
        logger.info("各章节 chunk 分布:")
        for ch, cnt in sorted(chapter_counts.items()):
            logger.info(f"  {ch}: {cnt} 个")


if __name__ == "__main__":
    main()
