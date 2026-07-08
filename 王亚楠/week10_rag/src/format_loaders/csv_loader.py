"""
CSV 格式加载器

处理策略：
  - 自动检测编码（UTF-8 → GBK）
  - 读取整个 CSV 渲染为一个 markdown 表格块
  - 限制最多 5000 行（避免超大文件）
"""

import csv
import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument, ParsedBlock
from .base_loader import detect_encoding, table_to_markdown

logger = logging.getLogger(__name__)

MAX_ROWS = 5000


class CsvLoader:
    """CSV 文件加载器。"""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        logger.info(f"加载 CSV: {file_path.name}")

        encoding = detect_encoding(file_path)
        logger.info(f"  检测到编码: {encoding}")

        rows = []
        with open(file_path, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= MAX_ROWS:
                    logger.warning(f"  CSV 超过 {MAX_ROWS} 行，已截断")
                    rows.append([f"... (共 {MAX_ROWS}+ 行，仅显示前 {MAX_ROWS} 行)"])
                    break
                rows.append(row)

        if not rows:
            logger.warning(f"  CSV 文件为空: {file_path.name}")
            return ParsedDocument(meta=meta, source=str(file_path), blocks=[])

        # 如果第一行可能是标题，保持原样；否则插入默认标题
        md_table = table_to_markdown(rows)

        blocks = [
            ParsedBlock(
                block_type="table",
                content=md_table,
                page_num=0,
                section_path=[f"CSV: {file_path.name}"],
            )
        ]

        logger.info(f"  解析完成: {len(rows)} 行 → 1 个表格块")
        return ParsedDocument(meta=meta, source=str(file_path), blocks=blocks)
