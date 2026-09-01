#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_knowledge.py — 第 1 步：直接解析手头文件 → 分块 → 知识库

支持直接解析原始课件文件（零依赖解析 PPTX，PDF 需可选安装 pymupdf）：
  * .pptx  : PPTX 本质是 zip 包，用标准库 zipfile+XML 读取内部 slide 的文本
             （等价于 python-pptx 的底层机制，但不需要安装任何库）
  * .pdf   : 逐页提取文本（需要 pymupdf：pip install pymupdf）
  * .txt/.md: 纯文本 / 带"===== [来源] 第N页 ====="标记的已提取文本

分块规则（对应课件 Part 2）：
  * PPTX 每页 / PDF 每页 是一个天然单元
  * 单元内按段落累积，超过 700 字符强制切块（块太大检索不精准）
  * 每块保留来源与页码，用于回答时的溯源引用

用法:
  python build_knowledge.py                            # 默认直接解析 data/RAG.pptx + data/ragas.pdf
  python build_knowledge.py --input 任意笔记.pptx      # 换成你自己的 PPT
  python build_knowledge.py --input 年报.pdf           # 换 PDF（需 pymupdf）
  python build_knowledge.py --input 任意笔记.txt data/别的.md
"""
import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
OUT_PATH  = DATA_DIR / "knowledge.json"
DEFAULT_INPUTS = [DATA_DIR / "RAG.pptx", DATA_DIR / "ragas.pdf"]   # 课件原件

MAX_CHUNK = 700          # 单块最大字符数
NOISE     = ("八斗学院出品", "盗版必究")   # 课件页脚噪声
MARKER    = re.compile(r"^=====\s*\[(.+?)\]\s*第\s*(\d+)\s*页?\s*=====\s*$")


def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s in NOISE:
        return True
    if re.fullmatch(r"\d{1,3}", s):          # 纯页码/节号
        return True
    return False


def extract_pptx(path: Path) -> list[dict]:
    """PPTX = zip 包：读 ppt/slides/slideN.xml 里所有 <a:t> 文本。零第三方依赖。"""
    NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    z = zipfile.ZipFile(path)
    slides = sorted(
        (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
        key=lambda n: int(re.search(r"(\d+)", n).group(1)),
    )
    if not slides:
        sys.exit(f"[错误] {path.name} 中找不到 ppt/slides/slideN.xml，不是有效 PPTX？")
    units = []
    for i, name in enumerate(slides, 1):
        root = ET.fromstring(z.read(name))
        paras = [t.text.strip() for t in root.iter(NS + "t") if t.text and t.text.strip()]
        units.append({"source": path.name, "page": i, "paras": [p for p in paras if not is_noise(p)]})
    return units


def extract_pdf(path: Path) -> list[dict]:
    try:
        import pymupdf
    except ImportError:
        sys.exit(f"[错误] 读取 PDF 需要 pymupdf：pip install pymupdf\n      （或先把 {path.name} 转成 txt 再 --input）")
    units = []
    doc = pymupdf.open(str(path))
    for i, page in enumerate(doc, 1):
        paras = [p.strip() for p in page.get_text().splitlines() if p.strip()]
        units.append({"source": path.name, "page": i, "paras": [p for p in paras if not is_noise(p)]})
    return units


def extract_txt(path: Path) -> list[dict]:
    """纯文本：带 ====[来源]第N页==== 标记时按页切；否则整文件算一页。"""
    units, cur = [], None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = MARKER.match(raw.strip())
        if m:
            if cur and cur["paras"]:
                units.append(cur)
            cur = {"source": m.group(1), "page": int(m.group(2)), "paras": []}
            continue
        if is_noise(raw):
            continue
        if cur is None:
            cur = {"source": path.name, "page": 1, "paras": []}
        cur["paras"].append(raw.strip())
    if cur and cur["paras"]:
        units.append(cur)
    return units


def chunk_units(units: list[dict]) -> list[dict]:
    """按来源单元分块：段落累积，超 MAX_CHUNK 即切。"""
    chunks = []
    for u in units:
        buf, buf_len = [], 0
        for para in u["paras"]:
            if buf_len + len(para) > MAX_CHUNK and buf:
                chunks.append({"content": "\n".join(buf)})
                buf, buf_len = [], 0
            buf.append(para)
            buf_len += len(para) + 1
        if buf:
            chunks.append({"content": "\n".join(buf)})

    final = []
    for c in chunks:                       # 单个超长段落按句号二次切分
        if len(c["content"]) <= MAX_CHUNK:
            final.append(c)
            continue
        for part in re.split(r"(?<=[。！？；])", c["content"]):
            part = part.strip()
            if part:
                final.append({"content": part})
    return final


def build(paths: list[Path]) -> dict:
    kb, missing = [], []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
            continue
        if p.suffix.lower() == ".pptx":
            units = extract_pptx(p)
        elif p.suffix.lower() == ".pdf":
            units = extract_pdf(p)
        else:
            units = extract_txt(p)
        for u in units:
            for i, c in enumerate(chunk_units([u]), 1):
                kb.append({
                    "content": c["content"],
                    "source":  f"[{u['source']}] 第{u['page']}页",
                })

    kb = [c for c in kb if len(c["content"]) >= 30]      # 过滤过短碎片
    for i, c in enumerate(kb):
        c["id"] = f"chunk_{i:04d}"

    payload = {
        "source_files": [str(p) for p in paths],
        "total_chunks": len(kb),
        "chunks": kb,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    src = Counter(c["source"].split("]")[0] + "]" for c in kb)
    avg = sum(len(c["content"]) for c in kb) / max(len(kb), 1)
    print(f"[OK] 知识库已生成 → {OUT_PATH}")
    print(f"     总块数: {len(kb)} | 平均长度: {avg:.0f} 字符")
    print(f"     来源分布: " + ", ".join(f"{k} {v}块" for k, v in src.most_common()))
    if missing:
        print(f"     [跳过] 不存在的输入: {missing}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="直接解析手头文件（PPTX/PDF/TXT/MD）并分块，生成知识库 JSON")
    ap.add_argument("--input", nargs="*", type=Path, default=None,
                    help="输入文件列表（默认: data/RAG.pptx data/ragas.pdf）")
    args = ap.parse_args()
    inputs = args.input or DEFAULT_INPUTS
    if not any(p.exists() for p in inputs):
        sys.exit(f"[错误] 找不到默认输入（{', '.join(map(str, inputs))}），请用 --input 指定文件")
    build(inputs)
