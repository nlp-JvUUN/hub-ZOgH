"""
文档加载器基类和共享工具函数

提供：
  - DocumentLoader Protocol：所有格式加载器的接口约定
  - 共享的标题/噪音检测、Markdown 表格转换等工具函数
    （从 parse_pdf.py 提取，保持逻辑完全一致）
"""

import re
from pathlib import Path
from typing import Protocol, Optional

from parsed_block_schema import ParsedDocument, ParsedBlock


# ── Protocol 定义 ───────────────────────────────────────────────────────────────

class DocumentLoader(Protocol):
    """
    文档加载器接口（结构化子类型，无需显式继承）。

    每个格式加载器只需实现两个方法：
      supports(file_path) -> bool    判断是否能处理该文件
      load(file_path, meta) -> ParsedDocument   加载并解析为统一结构
    """
    def supports(self, file_path: Path) -> bool: ...
    def load(self, file_path: Path, meta: dict) -> ParsedDocument: ...


# ── 共享工具函数 ────────────────────────────────────────────────────────────────

# 年报/文档里常见的章节标题模式
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),     # 第一章、第三节
    re.compile(r"^第[一二三四五六七八九十百]+条"),         # 第一条（法律/规范文档）
    re.compile(r"^[一二三四五六七八九十]、"),               # 一、二、
    re.compile(r"^\d+\.\s"),                                # 1. 2.
    re.compile(r"^\d+\)\s"),                                # 1) 2)
    re.compile(r"^[（(]\d+[)）]"),                           # (1) (2)
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,40}年度报告\s*$"),    # 页眉：公司名+年度报告
    re.compile(r"^\d+\s*$"),                # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),          # — 38 —
]


def is_noise_line(line: str) -> bool:
    """判断一行是否为噪音（页眉、页脚、独立页码等）。"""
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    判断一行是否是标题。
    有字体信息时用字体大小，没有时用文字规律（适用于 TXT/Markdown 等纯文本格式）。
    """
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(line.strip()) < 50:
        return True
    return any(p.match(line.strip()) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    """
    把二维表格数据转成 markdown 格式，方便 LLM 理解。
    与 parse_pdf.py 中的实现完全一致。
    """
    if not table:
        return ""

    # 清洗单元格：None 变空字符串，去掉换行
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    # 构建 markdown 表格
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 对齐列数（有些表格行列不整齐）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_encoding(file_path: Path) -> str:
    """
    检测文本文件编码。

    对于中文文档，按 UTF-8 → GBK → GB2312 → latin-1 的顺序尝试。
    如果 chardet 可用则优先使用 chardet（更准确），否则用简单 try-chain。
    """
    # 先读原始字节
    raw = file_path.read_bytes()

    # 尝试用 chardet
    try:
        import chardet
        result = chardet.detect(raw)
        if result and result.get("encoding"):
            enc = result["encoding"]
            if enc.lower() in ("gb2312", "gbk", "gb18030"):
                return "gbk"   # 统一用 GBK（GB2312 的超集）
            if enc.lower() in ("utf-8", "utf8"):
                return "utf-8"
            return enc.lower()
    except ImportError:
        pass

    # 简单 try-chain
    for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue

    # 最终兜底
    return "utf-8"
