"""
格式加载器注册表和调度函数

新增文件格式只需：
  1. 创建一个 loader 类，实现 supports() 和 load() 方法
  2. 在 EXTENSION_MAP 中注册扩展名
  3. 在 _LAZY_IMPORTS 中添加懒加载函数
"""

from pathlib import Path
from typing import Optional


# ── 扩展名 → loader 类名的映射 ──────────────────────────────────────────────────

EXTENSION_MAP: dict[str, str] = {
    ".pdf":  "PdfLoader",
    ".txt":  "TxtLoader",
    ".docx": "DocxLoader",
    ".doc":  "DocxLoader",
    ".md":   "MarkdownLoader",
    ".html": "HtmlLoader",
    ".htm":  "HtmlLoader",
    ".csv":  "CsvLoader",
    ".xlsx": "XlsxLoader",
    ".xls":  "XlsxLoader",
    ".png":  "ImageLoader",
    ".jpg":  "ImageLoader",
    ".jpeg": "ImageLoader",
    ".bmp":  "ImageLoader",
    ".tiff": "ImageLoader",
    ".tif":  "ImageLoader",
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())


# ── 懒加载：只在首次使用时导入具体 loader，避免不必要的依赖检查 ────────────────

_LOADER_CACHE: dict[str, type] = {}


def _load_pdf_loader():
    from .pdf_loader import PdfLoader
    return PdfLoader

def _load_txt_loader():
    from .txt_loader import TxtLoader
    return TxtLoader

def _load_docx_loader():
    from .docx_loader import DocxLoader
    return DocxLoader

def _load_markdown_loader():
    from .markdown_loader import MarkdownLoader
    return MarkdownLoader

def _load_html_loader():
    from .html_loader import HtmlLoader
    return HtmlLoader

def _load_csv_loader():
    from .csv_loader import CsvLoader
    return CsvLoader

def _load_xlsx_loader():
    from .xlsx_loader import XlsxLoader
    return XlsxLoader

def _load_image_loader():
    from .image_loader import ImageLoader
    return ImageLoader


_LAZY_IMPORTS: dict[str, callable] = {
    "PdfLoader":       _load_pdf_loader,
    "TxtLoader":       _load_txt_loader,
    "DocxLoader":      _load_docx_loader,
    "MarkdownLoader":  _load_markdown_loader,
    "HtmlLoader":      _load_html_loader,
    "CsvLoader":       _load_csv_loader,
    "XlsxLoader":      _load_xlsx_loader,
    "ImageLoader":     _load_image_loader,
}


def get_loader(file_path: Path) -> Optional[object]:
    """
    根据文件扩展名获取对应的 loader 实例。

    返回 None 表示不支持的文件类型。
    loader 使用懒加载——只有真正需要时才会导入对应的模块。
    """
    ext = file_path.suffix.lower()
    loader_name = EXTENSION_MAP.get(ext)
    if not loader_name:
        return None

    # 检查缓存
    if loader_name in _LOADER_CACHE:
        return _LOADER_CACHE[loader_name]()

    # 懒加载
    factory = _LAZY_IMPORTS.get(loader_name)
    if not factory:
        return None

    loader_cls = factory()
    _LOADER_CACHE[loader_name] = loader_cls
    return loader_cls()


def list_supported_extensions() -> list[str]:
    """返回所有支持的文件扩展名列表。"""
    return sorted(EXTENSION_MAP.keys())
