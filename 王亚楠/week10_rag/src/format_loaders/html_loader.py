"""
HTML 格式加载器

依赖：beautifulsoup4

处理策略：
  - <h1>-<h6> → title 块
  - <table> → markdown table 块
  - <p>/<div>/<li>/<blockquote> → text 块
  - 移除 <script>/<style>/<nav>/<footer>
"""

import logging
from pathlib import Path

from parsed_block_schema import ParsedDocument, ParsedBlock
from .base_loader import table_to_markdown

logger = logging.getLogger(__name__)

# 需要跳过的标签
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"}
# 标题标签
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# 块级标签（文字内容来源）
BLOCK_TAGS = {"p", "div", "li", "blockquote", "section", "article", "pre", "td", "th", "span"}


class HtmlLoader:
    """HTML 文件加载器。"""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in (".html", ".htm")

    def load(self, file_path: Path, meta: dict) -> ParsedDocument:
        logger.info(f"加载 HTML: {file_path.name}")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "需要安装 beautifulsoup4 来加载 .html 文件:\n"
                "  pip install beautifulsoup4 lxml"
            )

        html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        blocks = []
        section_stack: list[str] = []

        # 遍历 body 的直接子元素（如果 body 存在）
        body = soup.body if soup.body else soup
        if body is None:
            logger.warning(f"  HTML 无内容: {file_path.name}")
            return ParsedDocument(meta=meta, source=str(file_path), blocks=[])

        for element in body.descendants:
            # 跳过非标签元素
            if not hasattr(element, "name") or element.name is None:
                continue

            tag = element.name.lower()

            # 跳过不需要的标签
            if tag in SKIP_TAGS:
                continue

            # 跳过嵌套在标题/表格内的元素（会被父元素处理）
            parent = getattr(element, "parent", None)
            if parent and hasattr(parent, "name") and parent.name:
                p_tag = parent.name.lower()
                if p_tag in HEADING_TAGS or p_tag == "table":
                    continue

            text = element.get_text(strip=True)
            if not text:
                continue

            # 标题
            if tag in HEADING_TAGS:
                level = int(tag[1])
                section_stack = section_stack[:level - 1]
                section_stack.append(text)

                blocks.append(ParsedBlock(
                    block_type="title",
                    content=text,
                    page_num=0,
                    section_path=list(section_stack),
                ))

            # 表格
            elif tag == "table":
                rows_data = []
                for row in element.find_all("tr"):
                    cells = [cell.get_text(strip=True).replace("\n", " ") for cell in row.find_all(["td", "th"])]
                    if cells:
                        rows_data.append(cells)

                if rows_data:
                    md_table = table_to_markdown(rows_data)
                    blocks.append(ParsedBlock(
                        block_type="table",
                        content=md_table,
                        page_num=0,
                        section_path=list(section_stack),
                    ))

            # 块级文本
            elif tag in BLOCK_TAGS and not any(
                hasattr(p, "name") and p.name and p.name.lower() in BLOCK_TAGS
                for p in getattr(element, "parents", [])[:1]
            ):
                # 避免重复：只取最内层块级元素
                blocks.append(ParsedBlock(
                    block_type="text",
                    content=text,
                    page_num=0,
                    section_path=list(section_stack),
                ))

        logger.info(f"  解析完成: {len(blocks)} 个块")
        return ParsedDocument(meta=meta, source=str(file_path), blocks=blocks)
