"""
图片 OCR 格式加载器

依赖：Pillow + pytesseract（已在 requirements.txt 中）

处理策略：
  - 用 Pillow 打开图片
  - 用 pytesseract 进行中英文 OCR
  - 输出为单个 text 块，标记 is_ocr=True

如果 OCR 不可用，输出带错误信息的块。
"""

import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument, ParsedBlock

logger = logging.getLogger(__name__)

# 支持图片后缀
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# 检查 OCR 可用性
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ImageLoader:
    """图片 OCR 加载器。"""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        logger.info(f"加载图片 (OCR): {file_path.name}")

        if not OCR_AVAILABLE:
            logger.warning("  pytesseract 不可用，跳过 OCR")
            block = ParsedBlock(
                block_type="text",
                content=f"[图片 OCR 不可用（未安装 pytesseract/tesseract），文件: {file_path.name}]",
                page_num=0,
                section_path=[],
                is_ocr=True,
            )
            return ParsedDocument(meta=meta, source=str(file_path), blocks=[block])

        try:
            img = Image.open(file_path)
            # 如果图片太大，先缩放（限制长边 3000px）
            max_dim = 3000
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"  图片已缩放: {img.size}")

            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            img.close()

            if not text.strip():
                text = f"[图片中未检测到文字: {file_path.name}]"

            block = ParsedBlock(
                block_type="text",
                content=text.strip(),
                page_num=0,
                section_path=[],
                is_ocr=True,
            )
            logger.info(f"  OCR 完成: {len(text)} 字符")
            return ParsedDocument(meta=meta, source=str(file_path), blocks=[block])

        except Exception as e:
            logger.error(f"  OCR 失败: {e}")
            block = ParsedBlock(
                block_type="text",
                content=f"[图片 OCR 失败: {e}，文件: {file_path.name}]",
                page_num=0,
                section_path=[],
                is_ocr=True,
            )
            return ParsedDocument(meta=meta, source=str(file_path), blocks=[block])
