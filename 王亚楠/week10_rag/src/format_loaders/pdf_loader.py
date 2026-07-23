"""
PDF 格式加载器

封装 parse_pdf.AnnualReportParser，对外暴露统一的 DocumentLoader 接口。
内部使用 pdfplumber + PyMuPDF + OCR 三工具组合解析 PDF。
"""

import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument
from .base_loader import DocumentLoader

logger = logging.getLogger(__name__)


class PdfLoader:
    """
    PDF 文档加载器。

    委托给 parse_pdf.py 中的 AnnualReportParser 完成实际解析，
    本类只负责协议适配：supports() 检测扩展名，load() 调用解析并返回 ParsedDocument。
    """

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        """
        加载并解析 PDF 文件。

        Args:
            file_path: PDF 文件路径
            meta: 元信息字典（stock_code, year, company_name 等）

        Returns:
            ParsedDocument 包含所有解析后的 blocks
        """
        # 延迟导入，避免 PDF 依赖影响其他 loader
        import sys
        import os

        # 确保 src/ 在 sys.path 中（从 format_loaders/ 子目录运行时也能找到 parse_pdf）
        src_dir = Path(__file__).parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from parse_pdf import AnnualReportParser

        parser = AnnualReportParser(file_path, meta=meta)
        blocks = parser.parse()

        return ParsedDocument(
            meta=meta,
            source=str(file_path),
            blocks=blocks,
        )
