"""
多格式文档解析——统一入口

功能：
  1. 扫描 raw 目录中的所有支持文件格式
  2. 根据扩展名自动分配合适的加载器
  3. 统一输出 ParsedDocument JSON 到 data/parsed/
  4. 完全向后兼容——现有的 PDF 流水线不受影响

使用方式：
  # 扫描默认目录（data/raw_pdf/ + data/raw/）
  python src/parse_documents.py

  # 指定自定义目录
  python src/parse_documents.py --dir /path/to/documents

  # 仅处理特定格式
  python src/parse_documents.py --format pdf,txt

  # 使用 manifest.json 精确指定文件和元信息
  python src/parse_documents.py --manifest data/manifest.json

新增文件格式：
  在 format_loaders/ 目录下创建新的 loader，然后在
  format_loaders/__init__.py 中注册即可，无需修改本文件。
"""

import json
import sys
import logging
import argparse
from pathlib import Path

# 确保 src/ 在 path 中
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parsed_block_schema import ParsedDocument, save_parsed_document
from format_loaders import get_loader, SUPPORTED_EXTENSIONS, list_supported_extensions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdf"
PARSED_DIR  = BASE_DIR / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def scan_directory(dir_path: Path, format_filter: set = None) -> list[Path]:
    """
    扫描目录中的所有支持文件。

    Args:
        dir_path: 要扫描的目录
        format_filter: 可选的文件扩展名白名单（如 {'.pdf', '.txt'}）

    Returns:
        支持的文件路径列表
    """
    if not dir_path.exists():
        return []

    files = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        if format_filter and ext not in format_filter:
            continue
        for f in dir_path.glob(f"*{ext}"):
            if f.is_file():
                files.append(f)
    return sorted(files)


def parse_manifest(manifest_path: Path) -> list[dict]:
    """
    读取 manifest.json 并返回条目列表。

    每个条目格式：
      {
        "filename": "600519_2023_xxx.pdf",
        "stock_code": "600519",
        "year": "2023",
        "company_name": "贵州茅台",
        "file_type": "pdf"  (可选，从扩展名推断)
      }
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = []
    for item in manifest:
        filename = item.get("filename", "")
        if not filename:
            continue
        # 如果没有 file_type，从扩展名推断
        if "file_type" not in item:
            ext = Path(filename).suffix.lower().lstrip(".")
            item["file_type"] = ext
        entries.append(item)
    return entries


def process_file(file_path: Path, meta: dict = None, output_dir: Path | None = None) -> ParsedDocument | None:
    """
    处理单个文件：获取对应 loader，解析，保存。

    Args:
        file_path: 文件路径
        meta: 可选的元信息字典
        output_dir: 输出目录，默认使用 PARSED_DIR

    Returns:
        ParsedDocument 如果成功，None 如果格式不支持或解析失败
    """
    if meta is None:
        meta = {}
    if output_dir is None:
        output_dir = PARSED_DIR

    # 确保有基本元信息
    meta.setdefault("filename", file_path.name)
    if "stock_code" not in meta and "company_name" not in meta:
        # 尝试从文件名推断（如 "600519_2023_贵州茅台_xxx.pdf"）
        parts = file_path.stem.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            meta["stock_code"] = parts[0]
            meta["year"] = parts[1] if parts[1].isdigit() else ""
            if len(parts) >= 3:
                meta["company_name"] = parts[2]

    # 获取 loader
    loader = get_loader(file_path)
    if loader is None:
        ext = file_path.suffix.lower()
        logger.warning(f"不支持的文件格式: {ext} ({file_path.name})")
        return None

    logger.info(f"处理: {file_path.name} [{file_path.suffix.lower()}]")

    try:
        doc = loader.load(file_path, meta)
        out_path = save_parsed_document(doc, output_dir)
        logger.info(f"  → 已保存: {out_path.name}")
        return doc
    except ImportError as e:
        logger.error(f"  缺少依赖: {e}")
        return None
    except Exception as e:
        logger.error(f"  解析失败: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="多格式文档解析——统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
    )
    parser.add_argument("--dir", type=str, default=None,
                        help="指定输入目录（默认扫描 data/raw/ 和 data/raw_pdf/）")
    parser.add_argument("--manifest", type=str, default=None,
                        help="使用 manifest.json 精确指定文件列表和元信息")
    parser.add_argument("--format", type=str, default=None,
                        help="仅处理指定格式（逗号分隔，如 pdf,txt,docx）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 data/parsed/）")
    args = parser.parse_args()

    # 构建格式过滤器
    format_filter = None
    if args.format:
        format_filter = set()
        for f in args.format.split(","):
            ext = f".{f.strip().lstrip('.').lower()}"
            if ext in SUPPORTED_EXTENSIONS:
                format_filter.add(ext)
            else:
                logger.warning(f"忽略不支持的格式: {f}")

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PARSED_DIR

    logger.info("=" * 60)
    logger.info("多格式文档解析")
    logger.info(f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    logger.info("=" * 60)

    processed = 0
    failed = 0

    # 模式 1: 使用 manifest.json
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            logger.error(f"manifest 文件不存在: {manifest_path}")
            return

        entries = parse_manifest(manifest_path)
        logger.info(f"从 manifest 加载 {len(entries)} 个条目")

        for entry in entries:
            filename = entry["filename"]
            # 尝试在 raw 目录中查找文件
            found = None
            for search_dir in [RAW_DIR, RAW_PDF_DIR]:
                candidate = search_dir / filename
                if candidate.exists():
                    found = candidate
                    break

            if found is None:
                logger.warning(f"文件未找到: {filename}，跳过")
                failed += 1
                continue

            # 提取元信息
            meta = {
                "stock_code": entry.get("stock_code", ""),
                "year": entry.get("year", ""),
                "company_name": entry.get("company_name", ""),
                "filename": filename,
                "file_type": entry.get("file_type", ""),
            }
            # 保留 manifest 中的其他字段
            for key in ("plate", "title", "source_url", "announce_id"):
                if key in entry:
                    meta[key] = entry[key]

            if format_filter:
                ext = found.suffix.lower()
                if ext not in format_filter:
                    continue

            result = process_file(found, meta, output_dir)
            if result:
                processed += 1
            else:
                failed += 1

    # 模式 2: 扫描目录
    else:
        search_dirs = []
        if args.dir:
            search_dirs.append(Path(args.dir))
        else:
            search_dirs.extend([RAW_DIR, RAW_PDF_DIR])

        all_files = []
        for d in search_dirs:
            found = scan_directory(d, format_filter)
            logger.info(f"扫描 {d}: 找到 {len(found)} 个文件")
            all_files.extend(found)

        if not all_files:
            logger.warning("没有找到任何支持的文件")
            logger.info("请将文件放入 data/raw/ 目录，或使用 --dir 指定目录")
            return

        for file_path in all_files:
            result = process_file(file_path, output_dir=output_dir)
            if result:
                processed += 1
            else:
                failed += 1

    logger.info(f"\n完成: 成功 {processed} 个, 失败 {failed} 个")
    logger.info(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
