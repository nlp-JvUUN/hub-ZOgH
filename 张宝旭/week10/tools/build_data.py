# -*- coding: utf-8 -*-
"""
将 docx 解析进知识库（整库覆盖）。

用法：
  py -3 tools/build_data.py                  # 解析默认 docx
  py -3 tools/build_data.py path\to\new.docx
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC = ROOT / "整合版WAF-常见问题整理（自用）.docx"
KB_ROOT = ROOT / "kb"


def kb_paths(kb_id: str = "waf"):
    return KB_ROOT / kb_id / "entries.json", KB_ROOT / kb_id / "images"


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|#]+", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:80] or "item"


def iter_block_items(parent):
    from docx.document import Document as _Doc
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _Doc):
        elem = parent.element.body
    elif isinstance(parent, _Cell):
        elem = parent._tc
    else:
        raise ValueError("unsupported parent")

    for child in elem.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def save_image(image_part, save_dir: Path, used_names: set) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)
    orig = Path(image_part.partname).name
    name = orig
    i = 1
    while name in used_names:
        stem = Path(orig).stem
        suf = Path(orig).suffix
        name = f"{stem}_{i}{suf}"
        i += 1
    used_names.add(name)
    out_path = save_dir / name
    out_path.write_bytes(image_part.blob)
    return name


def paragraph_blocks(paragraph, save_dir: Path, used_image_names: set) -> list:
    blocks = []
    text_buf: list[str] = []

    def flush_text():
        if text_buf:
            t = "".join(text_buf).strip()
            if t:
                blocks.append({"type": "p", "text": t})
            text_buf.clear()

    for run in paragraph.runs:
        if run.text:
            text_buf.append(run.text)
        for drawing in run.element.findall(".//" + qn("w:drawing")):
            for blip in drawing.findall(".//" + qn("a:blip")):
                rId = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
                if not rId:
                    continue
                rel = paragraph.part.rels.get(rId)
                if rel is None or "image" not in rel.reltype:
                    continue
                fname = save_image(rel.target_part, save_dir, used_image_names)
                flush_text()
                blocks.append({"type": "img", "src": f"images/{fname}"})
    flush_text()
    return blocks


def table_block(table) -> dict:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            txt = "\n".join(p.text for p in cell.paragraphs).strip()
            cells.append(txt)
        rows.append(cells)
    return {"type": "table", "rows": rows}


def parse_docx(doc_path: Path, image_dir: Path) -> list[dict]:
    """解析 docx，返回 [{anchor, title, blocks}]，整库覆盖时直接交给 finalize。"""
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    doc = Document(str(doc_path))
    entries = []
    current = None
    used_image_names: set = set()
    seen_anchors: set = set()

    def make_anchor(title: str) -> str:
        base = "doc-" + slugify(title)
        anchor = base
        i = 2
        while anchor in seen_anchors:
            anchor = f"{base}-{i}"
            i += 1
        seen_anchors.add(anchor)
        return anchor

    for block in iter_block_items(doc):
        if block.__class__.__name__ == "Table":
            if current is not None:
                current["blocks"].append(table_block(block))
            continue
        p = block
        style = (p.style.name or "").strip()
        text = p.text.strip()
        if style == "Title":
            continue
        if style == "Heading 2":
            if not text:
                continue
            current = {
                "anchor": make_anchor(text),
                "title": text,
                "blocks": [],
            }
            entries.append(current)
            continue
        if current is None:
            continue
        if style == "Heading 4":
            if text:
                current["blocks"].append({"type": "h4", "text": text})
            for b in paragraph_blocks(p, image_dir, used_image_names):
                if b.get("type") == "img":
                    current["blocks"].append(b)
            continue
        sub_blocks = paragraph_blocks(p, image_dir, used_image_names)
        if not sub_blocks and not text:
            continue
        if sub_blocks and len(sub_blocks) == 1 and sub_blocks[0].get("type") == "p":
            runs = [r for r in p.runs if r.text and r.text.strip()]
            if runs and all(r.bold for r in runs) and len(text) <= 60:
                current["blocks"].append({"type": "h4", "text": text})
                continue
        current["blocks"].extend(sub_blocks)
    return entries


def finalize(entries: list[dict]) -> list[dict]:
    """补 id 和默认字段（tags），按出现顺序编号。"""
    out = []
    for i, d in enumerate(entries, start=1):
        d["id"] = i
        d.setdefault("tags", [])
        out.append(d)
    return out


def main():
    args = sys.argv[1:]
    kb_id = "waf"
    doc_path = DEFAULT_DOC
    # 解析参数：build_data.py [kb_id] [docx_path]
    if args:
        # 兼容只给 docx 路径的旧调用
        if args[0].lower().endswith(".docx"):
            doc_path = Path(args[0])
        else:
            kb_id = args[0]
            if len(args) > 1:
                doc_path = Path(args[1])

    if not doc_path.exists():
        print(f"[失败] 找不到 docx: {doc_path}")
        sys.exit(1)

    entries_json, img_dir = kb_paths(kb_id)
    entries_json.parent.mkdir(parents=True, exist_ok=True)
    if entries_json.exists():
        entries_json.unlink()

    parsed = parse_docx(doc_path, img_dir)
    merged = finalize(parsed)

    entries_json.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[整库覆盖] 库={kb_id} 共 {len(merged)} 条 -> {entries_json}")
    print(f"图片目录：{img_dir}")


if __name__ == "__main__":
    main()
