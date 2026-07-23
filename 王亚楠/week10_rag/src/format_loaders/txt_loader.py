"""
纯文本（.txt）格式加载器

处理策略：
  - 自动检测编码（UTF-8 → GBK → GB2312）
  - 按双换行符分隔段落
  - 用章节正则模式检测标题
  - 不支持表格（纯文本无表格结构）
"""

import re
import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument, ParsedBlock
from .base_loader import detect_encoding, is_title_line, is_noise_line

logger = logging.getLogger(__name__)


class TxtLoader:
    """纯文本文件加载器。"""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        logger.info(f"加载 TXT: {file_path.name}")

        # 检测编码并读取
        encoding = detect_encoding(file_path)
        logger.info(f"  检测到编码: {encoding}")
        text = file_path.read_text(encoding=encoding)

        # 按段落分隔
        paragraphs = re.split(r"\n\s*\n", text)

        blocks = []
        section_stack: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 尝试识别标题
            lines = para.split("\n")
            first_line = lines[0].strip()

            if is_title_line(first_line, fontsize=None, is_bold=False):
                # 维护章节栈
                self._update_section(first_line, section_stack)
                blocks.append(ParsedBlock(
                    block_type="title",
                    content=first_line,
                    page_num=0,
                    section_path=list(section_stack),
                ))
                # 如果标题行后还有内容，作为正文
                remaining = "\n".join(lines[1:]).strip()
                if remaining and not is_noise_line(remaining):
                    blocks.append(ParsedBlock(
                        block_type="text",
                        content=remaining,
                        page_num=0,
                        section_path=list(section_stack),
                    ))
            else:
                if not is_noise_line(para):
                    blocks.append(ParsedBlock(
                        block_type="text",
                        content=para,
                        page_num=0,
                        section_path=list(section_stack),
                    ))

        logger.info(f"  解析完成: {len(blocks)} 个块")
        return ParsedDocument(meta=meta, source=str(file_path), blocks=blocks)

    @staticmethod
    def _update_section(title: str, stack: list[str]):
        """维护章节栈，与原 PDF 解析器逻辑一致。"""
        if re.match(r"^第[一二三四五六七八九十]+章", title):
            stack.clear()
            stack.append(title)
        elif re.match(r"^第[一二三四五六七八九十]+节", title):
            stack[:] = stack[:1] + [title]
        elif re.match(r"^[一二三四五六七八九十]、", title):
            stack[:] = stack[:2] + [title]
        else:
            stack.append(title)
