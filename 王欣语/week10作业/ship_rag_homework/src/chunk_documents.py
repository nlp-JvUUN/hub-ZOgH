"""
文档分块脚本：对解析后的船舶术语资料做分块处理

重点（三种分块策略的对比）：
  策略A  固定大小分块  —— 最简单，但会切断问答对
  策略B  语义分块      —— 以问答对为边界，保留完整性（默认）
  策略C  层级分块      —— 父块=主题分类，子块=单个问答对

企业级 RAG 通常用 B 或 C，
让学生先跑通 A，再体会 B/C 在召回效果上的区别。

输出格式说明：
  每个 chunk 是一个 dict，包含：
    - chunk_id      唯一标识
    - content       文本内容（供 embedding）
    - metadata      元信息（供过滤/溯源）
      - source_file  来源CSV文件名
      - category     术语类/问答类
      - doc_source   具体文档来源（如 GBT+7727-2025）
      - row_num      原始行号
      - block_type   qa_pair / term_def
      - strategy     分块策略名
"""

import json
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。
    缺点：无视问答对边界，问题和答案可能被切断。
    优点：实现最简单，块大小可预测。
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块（以问答对为边界）─────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
) -> Iterator[dict]:
    """
    按问答对分块：每个问答对尽量保持完整。
    如果单个问答对超过 max_chunk_size，再按固定大小切分。

    优点：保留语义完整性，问题和答案不被切断。
    缺点：块大小不均匀（长答案可能很大）。
    """
    for block in blocks:
        content = block.get("content", "")
        meta = block.get("metadata", {})

        # 如果内容不长，直接作为一个 chunk
        if len(content) <= max_chunk_size:
            yield {
                "content": content,
                "metadata": {
                    "source_file": meta.get("source_file", ""),
                    "category": meta.get("category", ""),
                    "doc_source": meta.get("doc_source", ""),
                    "row_num": meta.get("row_num", -1),
                    "block_type": block.get("block_type", "text"),
                    "section_path": block.get("section_path", []),
                }
            }
        else:
            # 长内容按固定大小切分，但保留元信息
            for i, text_chunk in enumerate(chunk_fixed(content, max_chunk_size, 50)):
                yield {
                    "content": text_chunk,
                    "metadata": {
                        "source_file": meta.get("source_file", ""),
                        "category": meta.get("category", ""),
                        "doc_source": meta.get("doc_source", ""),
                        "row_num": meta.get("row_num", -1),
                        "block_type": block.get("block_type", "text"),
                        "section_path": block.get("section_path", []),
                        "is_partial": True,
                        "part_index": i,
                    }
                }


# ── 策略 C：层级分块（父子块）─────────────────────────────────────────────────

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size: int = 400,
    overlap: int = 50,
) -> Iterator[dict]:
    """
    两级结构：
      父块（parent）：同一 doc_source 下的多个问答对聚合
      子块（child）：单个问答对或问答对的片段

    检索时：命中子块 → 取父块内容 → 给 LLM 读父块（更完整上下文）
    即 "Small-to-Big" 检索策略。
    """
    # 按 doc_source 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for block in blocks:
        doc_source = block.get("metadata", {}).get("doc_source", "未知")
        groups[doc_source].append(block)

    for doc_source, group_blocks in groups.items():
        # 构建父块内容
        parent_content = f"【来源：{doc_source}】\n\n"
        for b in group_blocks:
            parent_content += b.get("content", "") + "\n\n"

        parent_id = f"parent_{doc_source.replace(' ', '_').replace('/', '_')[:30]}"

        # 从父块中切子块
        full_text = parent_content
        start = 0
        child_idx = 0

        while start < len(full_text):
            end = min(start + child_size, len(full_text))
            child_content = full_text[start:end]

            yield {
                "content": child_content,
                "metadata": {
                    "parent_id": parent_id,
                    "parent_content": parent_content,  # 存全量供 LLM 读
                    "doc_source": doc_source,
                    "block_type": "text",
                    "section_path": ["层级分块", doc_source],
                    "child_index": child_idx,
                }
            }

            start += child_size - overlap
            child_idx += 1


# ── 主流程 ────────────────────────────────────────────────────────────────────

STRATEGY = "semantic"   # 改成 "fixed" 或 "hierarchical" 体验不同策略

def build_chunk_id(source_file: str, row_num: int, idx: int) -> str:
    """构建唯一 chunk_id。"""
    safe_name = Path(source_file).stem.replace(" ", "_")
    return f"{safe_name}_r{row_num:04d}_c{idx:04d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])
    meta = data.get("meta", {})

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    # 根据策略生成 chunks
    raw_chunks = []

    if strategy == "fixed":
        # 将所有 blocks 合并为大文本后切分
        full_text = "\n\n".join(b.get("content", "") for b in blocks)
        for i, text_chunk in enumerate(chunk_fixed(full_text)):
            raw_chunks.append({
                "content": text_chunk,
                "metadata": {
                    "source_file": parsed_path.name,
                    "block_type": "text",
                    "section_path": [],
                    "strategy": strategy,
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

    # 补充公共元信息并生成 chunk_id
    result = []
    for idx, chunk in enumerate(raw_chunks):
        meta_info = chunk.get("metadata", {})
        source_file = meta_info.get("source_file", parsed_path.name)
        row_num = meta_info.get("row_num", idx)

        chunk_id = build_chunk_id(source_file, row_num if row_num != -1 else idx, idx)
        chunk["chunk_id"] = chunk_id
        chunk["metadata"]["strategy"] = strategy
        chunk["metadata"]["source_file"] = source_file
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
        logger.error("没有找到解析结果，请先运行 parse_csv.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    # 合并所有 chunk 到一个文件，方便统一建索引
    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    # 简单统计
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    # 按 category 统计
    from collections import Counter
    categories = Counter(c["metadata"].get("category", "unknown") for c in all_chunks)
    for cat, count in categories.items():
        logger.info(f"  {cat}: {count} 个 chunk")


if __name__ == "__main__":
    main()
