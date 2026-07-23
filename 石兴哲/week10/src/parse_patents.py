"""
专利数据解析脚本：将 Google Patents 的 JSON 原始数据转为结构化 blocks

与 week10_rag 的 parse_pdf.py 对比：
  - 年报版：pdfplumber + PyMuPDF + Tesseract OCR 三件套，~300行
  - 专利版：纯 JSON → dict 映射，~100行
  - 这就是"数据源质量决定工程复杂度"的直观体现

专利文本的天然结构：
  title       → 发明名称（一句话）
  abstract    → 摘要（技术方案概述）
  claims      → 权利要求（逐条独立，每条是一个法律主张）
  description → 说明书（背景技术 → 发明内容 → 附图说明 → 具体实施方式）
  background  → 背景技术（提取自 description 的前部）
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
PARSED_DIR  = BASE_DIR / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 数据结构 ──────────────────────────────────────────────────────────────────
# 与 week10_rag 保持相同的 ParsedBlock 结构，保证下游代码兼容

@dataclass
class ParsedBlock:
    """
    专利文本的一个结构化块。

    与年报版 ParsedBlock 的差异：
      - 新增 block_type: "abstract"（摘要），"claim"（权利要求）
      - section_path 改为专利结构：["说明书", "具体实施方式"]
      - 去掉 is_ocr / raw_table 字段（专利 XML 不需要）
    """
    block_type:   str            # "title" | "abstract" | "claim" | "text"
    content:      str
    page_num:     int = 0        # 专利无页码概念，置 0
    section_path: list[str] = field(default_factory=list)
    claim_num:    int = 0        # 权利要求编号（仅 claim 类型使用）


# ── 专利文本预处理 ──────────────────────────────────────────────────────────────

# 说明书中的章节标题模式（中英文）
DESCRIPTION_SECTION_PATTERNS = [
    # 中文
    re.compile(r'^(技术领域|背景技术|发明内容|附图说明|具体实施方式|实施例|有益效果)'),
    re.compile(r'^[一二三四五六七八九十]+[、.)]'),
    # 英文
    re.compile(r'^(TECHNICAL FIELD|BACKGROUND|SUMMARY|DETAILED DESCRIPTION'
               r'|BRIEF DESCRIPTION|DESCRIPTION OF EMBODIMENTS|EXAMPLES?)',
               re.IGNORECASE),
    re.compile(r'^(Field of the Invention|Background of the Invention'
               r'|Summary of the Invention|Brief Description of the Drawings'
               r'|Detailed Description|Description of the Preferred Embodiment)',
               re.IGNORECASE),
]


def _classify_title(title: str) -> str:
    """判断标题属于专利的哪个章节。"""
    for pat in DESCRIPTION_SECTION_PATTERNS:
        if pat.match(title.strip()):
            return "section_title"
    return "title"


def _split_claims(claims: list[str]) -> list[dict]:
    """将权利要求按编号拆分，每一条独立。"""
    result = []
    claim_pattern = re.compile(
        r'(?:^|\n)\s*(?:Claim\s*)?(\d{1,3})[\.、:)\]]\s*',
        re.IGNORECASE
    )

    for claim_text in claims:
        # 尝试按编号拆分
        parts = claim_pattern.split(claim_text)
        if len(parts) >= 3:
            # 格式：["", "1", "text...", "2", "text..."]
            for i in range(1, len(parts), 2):
                num = int(parts[i])
                text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if text:
                    result.append({"num": num, "text": text})
        else:
            # 不能拆分的整体当做一条
            result.append({"num": len(result) + 1, "text": claim_text.strip()})

    return result


def _split_description(desc: str) -> list[dict]:
    """
    将说明书按章节拆分，返回 [{"section": "背景技术", "content": "..."}, ...]。
    """
    if not desc:
        return [{"section": "说明书", "content": ""}]

    # 尝试按标题拆分
    section_pattern = re.compile(
        r'(?:\n|^)\s*((?:技术领域|背景技术|发明内容|附图说明|具体实施方式|实施例|有益效果'
        r'|FIELD OF THE INVENTION|BACKGROUND(?: OF THE INVENTION)?'
        r'|SUMMARY(?: OF THE INVENTION)?'
        r'|BRIEF DESCRIPTION(?: OF THE DRAWINGS)?'
        r'|DETAILED DESCRIPTION(?: OF THE (?:PREFERRED|EMBODIMENTS|INVENTION))?'
        r'|DESCRIPTION OF (?:THE )?(?:PREFERRED )?EMBODIMENTS?'
        r'|EXAMPLES?'
        r'))[\s:：\n]+',
        re.IGNORECASE
    )

    parts = section_pattern.split(desc)
    if len(parts) <= 2:
        # 没有识别到章节标题，整段当做一个 section
        return [{"section": "说明书", "content": desc.strip()}]

    # 第一个元素是标题前的空白
    sections = []
    # parts 格式: [prefix, "背景技术", "content...", "发明内容", "content..."]
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append({"section": section_name, "content": content})

    if not sections:
        sections.append({"section": "说明书", "content": desc.strip()})

    return sections


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

def parse_patent(patent_data: dict) -> list[ParsedBlock]:
    """
    将一条专利的 JSON 数据转换为 ParsedBlock 列表。

    输出顺序：
      title → abstract → description sections → claims
    这个顺序恰好也是 RAG 检索的优先级建议。
    """
    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []

    title = patent_data.get("title", "")
    abstract = patent_data.get("abstract", "")
    description = patent_data.get("description", "")
    claims = patent_data.get("claims", [])

    # ── 1. 标题 ──
    if title:
        section_stack = [title]
        blocks.append(ParsedBlock(
            block_type="title",
            content=title.strip(),
            section_path=list(section_stack),
        ))

    # ── 2. 摘要 ──
    if abstract and len(abstract) > 20:
        blocks.append(ParsedBlock(
            block_type="abstract",
            content=abstract.strip(),
            section_path=list(section_stack) + ["摘要"],
        ))

    # ── 3. 说明书（按章节拆分）──
    if description:
        sections = _split_description(description)
        for sec in sections:
            sec_name = sec["section"]
            sec_text = sec["content"]
            if not sec_text:
                continue

            # 将章节内容按段落拆分为多个 "text" block
            paragraphs = _split_into_paragraphs(sec_text)
            for para in paragraphs:
                if not para.strip():
                    continue
                blocks.append(ParsedBlock(
                    block_type="text",
                    content=para.strip(),
                    section_path=list(section_stack) + [sec_name],
                ))
    else:
        # 没有说明书文本时标记为空
        logger.debug(f"  专利 {patent_data.get('patent_id', '?')}: 无说明书文本")

    # ── 4. 权利要求（逐条独立）──
    if claims:
        claim_items = _split_claims(claims)
        for ci in claim_items:
            if ci["text"]:
                blocks.append(ParsedBlock(
                    block_type="claim",
                    content=ci["text"].strip(),
                    section_path=list(section_stack) + ["权利要求"],
                    claim_num=ci["num"],
                ))

    return blocks


def _split_into_paragraphs(text: str) -> list[str]:
    """将文本按段落拆分（双换行或单换行+缩进）。"""
    # 先按双换行拆分
    raw_paras = re.split(r'\n\s*\n', text)
    result = []
    for para in raw_paras:
        para = para.strip()
        # 过滤太短的行
        if len(para) < 10:
            continue
        # 过滤表格/代码块（专利中较少）
        if para.count('|') > 5:
            continue
        result.append(para)
    return result


# ── 保存 ──────────────────────────────────────────────────────────────────────

def save_parsed(patent_id: str, meta: dict, blocks: list[ParsedBlock]):
    """保存解析结果。"""
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', patent_id)
    out_path = PARSED_DIR / f"{safe_name}.json"

    output = {
        "meta": meta,
        "source": patent_id,
        "blocks": [asdict(b) for b in blocks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    block_types = {}
    for b in blocks:
        block_types[b.block_type] = block_types.get(b.block_type, 0) + 1
    logger.info(
        f"  → {out_path.name}  ({len(blocks)} blocks: "
        + ", ".join(f"{k}={v}" for k, v in sorted(block_types.items()))
        + ")"
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = BASE_DIR / "data" / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        # 没有 manifest 就直接扫 raw/ 目录
        manifest = []
        for p in RAW_DIR.glob("*.json"):
            with open(p, encoding="utf-8") as f:
                manifest.append(json.load(f))

    if not manifest:
        logger.error("没有找到任何专利数据，请先运行 fetch_patents.py")
        return

    logger.info(f"共 {len(manifest)} 份专利待解析")

    total_blocks = 0
    for item in manifest:
        pid = item.get("patent_id", "")
        filename = item.get("filename", "")
        if not pid:
            continue

        # 从 raw/ 目录加载完整的专利数据（manifest 只有元数据）
        raw_path = RAW_DIR / filename
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                patent_data = json.load(f)
        else:
            # fallback: 尝试用 patent_id 拼接文件名
            alt_path = RAW_DIR / f"{pid}.json"
            if alt_path.exists():
                with open(alt_path, encoding="utf-8") as f:
                    patent_data = json.load(f)
            else:
                logger.warning(f"  找不到原始数据文件: {filename}，跳过")
                continue

        logger.info(f"解析: {pid}  — {patent_data.get('title', '')[:50]}")

        try:
            blocks = parse_patent(patent_data)  # 传完整数据而非 manifest item
            total_blocks += len(blocks)
            save_parsed(pid, {
                "patent_id":         pid,
                "title":             patent_data.get("title", ""),
                "assignee":          patent_data.get("assignee", ""),
                "patent_office":     patent_data.get("patent_office", ""),
                "publication_date":  patent_data.get("publication_date", ""),
                "inventors":         patent_data.get("inventors", []),
            }, blocks)
        except Exception as e:
            logger.error(f"  解析失败: {e}")

    logger.info(f"\n全部解析完成，共 {total_blocks} 个 blocks → {PARSED_DIR}")


if __name__ == "__main__":
    main()
