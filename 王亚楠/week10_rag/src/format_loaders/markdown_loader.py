"""
Markdown（.md）格式加载器

处理策略：
  - 纯正则解析（无需第三方 Markdown 库）
  - # 标题 → title 块
  - | 表格 → table 块
  - 代码块引用块 → text 块
  - 普通段落 → text 块
"""

import re
import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument, ParsedBlock
from .base_loader import table_to_markdown

logger = logging.getLogger(__name__)

# 匹配 markdown 标题
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# 匹配 markdown 表格行（至少包含一个 |）
TABLE_LINE_RE = re.compile(r"^\|.+\|.*\|?\s*$")
# 匹配表格分隔行
TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")
# 代码块起止
CODE_FENCE_RE = re.compile(r"^```")


class MarkdownLoader:
    """Markdown 文件加载器。"""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".md"

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        logger.info(f"加载 Markdown: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")

        blocks = []
        section_stack: list[str] = []

        # 将文档按空行分隔为段落块
        paragraphs = re.split(r"\n\s*\n", text)

        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if not para:
                i += 1
                continue

            lines = para.split("\n")

            # 检测标题
            heading_match = HEADING_RE.match(lines[0])
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # 维护章节栈
                section_stack = section_stack[:level - 1]
                section_stack.append(title)

                blocks.append(ParsedBlock(
                    block_type="title",
                    content=title,
                    page_num=0,
                    section_path=list(section_stack),
                ))

                # 标题行后面可能还有正文
                remaining = "\n".join(lines[1:]).strip()
                if remaining:
                    blocks.append(ParsedBlock(
                        block_type="text",
                        content=remaining,
                        page_num=0,
                        section_path=list(section_stack),
                    ))
                i += 1
                continue

            # 检测表格（多行连续的 | 行）
            if TABLE_LINE_RE.match(lines[0]):
                table_lines = []
                while i < len(paragraphs):
                    p = paragraphs[i].strip()
                    p_lines = p.split("\n")
                    all_table = all(TABLE_LINE_RE.match(l) or TABLE_SEP_RE.match(l) for l in p_lines if l.strip())
                    if all_table:
                        table_lines.extend(p_lines)
                        i += 1
                    else:
                        break

                # 跳过表格分隔行，解析为二维数组
                data_rows = []
                for line in table_lines:
                    if TABLE_SEP_RE.match(line):
                        continue
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    data_rows.append(cells)

                if data_rows:
                    md_table = table_to_markdown(data_rows)
                    blocks.append(ParsedBlock(
                        block_type="table",
                        content=md_table,
                        page_num=0,
                        section_path=list(section_stack),
                    ))
                continue

            # 检测代码块
            if CODE_FENCE_RE.match(lines[0]):
                code_lines = [para]
                i += 1
                while i < len(paragraphs):
                    p = paragraphs[i]
                    code_lines.append(p)
                    if CODE_FENCE_RE.match(p.strip()):
                        i += 1
                        break
                    i += 1
                blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n\n".join(code_lines),
                    page_num=0,
                    section_path=list(section_stack),
                ))
                continue

            # 普通段落
            blocks.append(ParsedBlock(
                block_type="text",
                content=para,
                page_num=0,
                section_path=list(section_stack),
            ))
            i += 1

        logger.info(f"  解析完成: {len(blocks)} 个块")
        return ParsedDocument(meta=meta, source=str(file_path), blocks=blocks)
