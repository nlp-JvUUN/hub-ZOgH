"""
共享数据模型：所有格式加载器输出的统一结构

这个模块是解析层和分块层之间的"合约"——
只要 loader 输出的 JSON 符合 ParsedDocument 结构，
chunk_documents.py 就能无差别处理。

设计原则：
  - 零外部依赖（只有 dataclasses + pathlib）
  - 不依赖任何 PDF 库，因此 TXT/DOCX 等 loader 可以独立运行
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ParsedBlock:
    """
    一个解析块 = 文档里的一段连续内容（文字段落 / 表格 / 标题）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户来源位置。
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int            # 页码（无分页的格式用 0）
    section_path: list[str] = field(default_factory=list)  # ["第三章", "一、概述"]
    is_ocr:       bool     = False  # 是否经过 OCR，质量可能较低
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


@dataclass
class ParsedDocument:
    """
    一个解析后的文档 = 元信息 + 源文件路径 + 解析块列表

    序列化为 JSON 后存入 data/parsed/，供 chunk_documents.py 消费。
    """
    meta:   dict              # {"stock_code": "600519", "year": "2023", ...}
    source: str               # 源文件绝对路径
    blocks: list[ParsedBlock] = field(default_factory=list)


def parsed_document_to_dict(doc: ParsedDocument) -> dict:
    """将 ParsedDocument 转为可 JSON 序列化的 dict。"""
    return {
        "meta":   doc.meta,
        "source": doc.source,
        "blocks": [asdict(b) for b in doc.blocks],
    }


def save_parsed_document(doc: ParsedDocument, output_dir: Path):
    """
    将解析结果保存为 JSON 文件。
    文件名 = 源文件 stem + .json，保存到 output_dir。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem     = Path(doc.source).stem
    out_path = output_dir / f"{stem}.json"

    output = parsed_document_to_dict(doc)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return out_path
