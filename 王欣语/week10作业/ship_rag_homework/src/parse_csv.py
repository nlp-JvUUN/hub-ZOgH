"""
CSV 解析脚本：将船舶术语训练资料转换为结构化 JSON

重点：
  1. 处理不同格式的 CSV（术语类 vs 问答类）
  2. 统一输出为 ParsedBlock 格式，与 PDF 解析输出兼容
  3. 保留完整的元信息供后续检索和溯源

输入：
  data/raw_csv/术语类120个随机问题评分.csv
  data/raw_csv/问答对.csv

输出：
  data/parsed/术语类120个随机问题评分.json
  data/parsed/问答对.json
  data/manifest.json

每个 block 结构：
  {
    "block_type": "qa_pair",
    "content": "问题：...\n答案：...",
    "page_num": 行号,
    "section_path": ["术语类", "GBT+7727-2025"],
    "is_ocr": false,
    "metadata": {
      "source_file": "...",
      "category": "术语类/问答类",
      "row_num": 1,
      "doc_source": "GBT+7727-2025 船舶通用术语"
    }
  }
"""

import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_csv"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    block_type: str      # "qa_pair" | "term_def" | "title"
    content: str
    page_num: int        # 对应 CSV 行号
    section_path: list    # [大类, 来源文件/章节]
    is_ocr: bool = False
    metadata: dict = None


# ── 解析术语类 CSV ────────────────────────────────────────────────────────────

def parse_term_csv(csv_path: Path) -> list[ParsedBlock]:
    """
    解析术语类 CSV，格式：
      文件名,问题,答案,打分1-10

    注意：CSV 可能有 BOM（\ufeff），需要处理。
    """
    blocks = []
    category = "术语类"

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头

        for row_num, row in enumerate(reader, start=2):
            if len(row) < 3:
                continue

            source_file = row[0].strip() if len(row) > 0 else ""
            question    = row[1].strip() if len(row) > 1 else ""
            answer      = row[2].strip() if len(row) > 2 else ""

            if not question or not answer:
                continue

            # 从文件名推断章节/标准来源
            doc_source = source_file.replace(".csv", "") if source_file else "未知来源"
            section_path = [category, doc_source]

            # 构建内容：问题 + 答案
            content = f"问题：{question}\n答案：{answer}"

            block = ParsedBlock(
                block_type="qa_pair",
                content=content,
                page_num=row_num,
                section_path=section_path,
                metadata={
                    "source_file": csv_path.name,
                    "category": category,
                    "row_num": row_num,
                    "doc_source": doc_source,
                    "question": question,
                    "answer": answer,
                    "score": row[3].strip() if len(row) > 3 else "",
                }
            )
            blocks.append(block)

    logger.info(f"  解析术语类: {len(blocks)} 条问答对")
    return blocks


# ── 解析问答对 CSV ────────────────────────────────────────────────────────────

def parse_qa_csv(csv_path: Path) -> list[ParsedBlock]:
    """
    解析问答对 CSV，格式：
      文件名,问题,回答,评分,说明评分原因

    注意：此 CSV 使用 | 作为分隔符（从代码片段推断）。
    """
    blocks = []
    category = "问答类"

    with open(csv_path, encoding="utf-8-sig") as f:
        # 先检测分隔符
        sample = f.read(4096)
        f.seek(0)

        delimiter = ","
        if sample.count("|") > sample.count(","):
            delimiter = "|"
            logger.info(f"  检测到 '|' 分隔符")

        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)  # 跳过表头

        for row_num, row in enumerate(reader, start=2):
            if len(row) < 3:
                continue

            source_file = row[0].strip() if len(row) > 0 else ""
            question    = row[1].strip() if len(row) > 1 else ""
            answer      = row[2].strip() if len(row) > 2 else ""

            if not question or not answer:
                continue

            doc_source = source_file.replace(".pdf", "").replace(".csv", "") if source_file else "未知来源"
            section_path = [category, doc_source]

            content = f"问题：{question}\n答案：{answer}"

            block = ParsedBlock(
                block_type="qa_pair",
                content=content,
                page_num=row_num,
                section_path=section_path,
                metadata={
                    "source_file": csv_path.name,
                    "category": category,
                    "row_num": row_num,
                    "doc_source": doc_source,
                    "question": question,
                    "answer": answer,
                    "score": row[3].strip() if len(row) > 3 else "",
                    "score_reason": row[4].strip() if len(row) > 4 else "",
                }
            )
            blocks.append(block)

    logger.info(f"  解析问答类: {len(blocks)} 条问答对")
    return blocks


# ── 保存解析结果 ────────────────────────────────────────────────────────────────

def save_parsed(blocks: list[ParsedBlock], output_name: str, meta: dict = None):
    """将解析结果保存为 JSON，与 PDF 解析输出格式兼容。"""
    out_path = PARSED_DIR / f"{output_name}.json"

    output = {
        "meta": meta or {},
        "source": "CSV",
        "blocks": [asdict(b) for b in blocks],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"  已保存 → {out_path}")
    return out_path


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    all_manifest = []
    total_blocks = 0

    # 1. 解析术语类
    term_csv = RAW_DIR / "术语类120个随机问题评分.csv"
    if term_csv.exists():
        logger.info(f"解析: {term_csv.name}")
        term_blocks = parse_term_csv(term_csv)
        save_parsed(term_blocks, "术语类120个随机问题评分", meta={
            "category": "术语类",
            "total_qa_pairs": len(term_blocks),
            "source_csv": term_csv.name,
        })
        all_manifest.append({
            "filename": "术语类120个随机问题评分.json",
            "category": "术语类",
            "blocks": len(term_blocks),
        })
        total_blocks += len(term_blocks)
    else:
        logger.warning(f"找不到: {term_csv}")

    # 2. 解析问答对
    qa_csv = RAW_DIR / "问答对.csv"
    if qa_csv.exists():
        logger.info(f"解析: {qa_csv.name}")
        qa_blocks = parse_qa_csv(qa_csv)
        save_parsed(qa_blocks, "问答对", meta={
            "category": "问答类",
            "total_qa_pairs": len(qa_blocks),
            "source_csv": qa_csv.name,
        })
        all_manifest.append({
            "filename": "问答对.json",
            "category": "问答类",
            "blocks": len(qa_blocks),
        })
        total_blocks += len(qa_blocks)
    else:
        logger.warning(f"找不到: {qa_csv}")

    # 3. 保存 manifest
    manifest = {
        "project": "船舶术语RAG训练资料",
        "total_blocks": total_blocks,
        "files": all_manifest,
    }
    manifest_path = Path(__file__).parent.parent / "data" / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest 已保存 → {manifest_path}")

    logger.info(f"\n全部解析完成！共 {total_blocks} 条问答对")


if __name__ == "__main__":
    main()
