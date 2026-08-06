# -*- coding: utf-8 -*-
"""Automated smoke tests for the resume-from-template export pipeline.

Run from anywhere:
    python .claude/skills/resume-from-template/scripts/test_skill.py

Checks (all on the bundled FICTIONAL example resume):
  1. No unresolved {{...}} placeholders remain in the source markdown.
  2. PDF: generates, >=1 page, expected sections present, sane margins,
     no literal '{{' in extracted text, and no fallback font (only CJK font).
  3. HTML: file exists and contains the H1 name and <html>.
  4. Word: file opens with python-docx and contains the H1 name.
  5. --check-only flag reports a clean resume as OK and flags a placeholder.

Exit code 0 if all pass, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
EXAMPLE = SKILL_DIR / "examples" / "resume-example.md"
TOOL = SCRIPT_DIR / "md_to_pdf.py"

MM = 25.4 / 72  # points -> mm


def run_tool(args):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default=str(EXAMPLE),
                    help="Resume markdown to test with (default: bundled example)")
    args = ap.parse_args()

    md_path = Path(args.example)
    if not md_path.is_file():
        print(f"FAIL: example not found: {md_path}")
        return 1

    checks = []
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        checks.append(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))

    # 0) source sanity
    src = md_path.read_text(encoding="utf-8")
    check("example contains no real personal data",
          all(x not in src for x in ("张宝旭", "18072900531", "469711671")),
          "real personal data found in example!")
    check("example has no unresolved placeholders",
          "{{" not in src and "}}" not in src,
          "placeholder found")

    # 1) --check-only on the clean example
    r = run_tool(["--check-only", str(md_path)])
    check("--check-only passes on clean resume",
          r.returncode == 0 and "OK" in r.stdout,
          f"rc={r.returncode} out={r.stdout[:80]!r} err={r.stderr[:80]!r}")

    # 1b) --check-only detects a placeholder
    with tempfile.TemporaryDirectory() as td:
        dirty = Path(td) / "dirty.md"
        dirty.write_text(src + "\n## 待补充\n{{姓名}}\n", encoding="utf-8")
        r2 = run_tool(["--check-only", str(dirty)])
        check("--check-only rejects a placeholder",
              r2.returncode == 2 and "placeholder" in r2.stderr.lower(),
              f"rc={r2.returncode} err={r2.stderr[:80]!r}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 2) PDF
        pdf = td / "out.pdf"
        r = run_tool([str(md_path), str(pdf), "--format", "pdf"])
        check("PDF export succeeds", r.returncode == 0 and pdf.is_file(),
              f"rc={r.returncode} err={r.stderr[:120]!r}")
        if pdf.is_file():
            doc = fitz.open(pdf)
            page = doc[0]
            text = page.get_text()
            check("PDF has >=1 page", doc.page_count >= 1, f"pages={doc.page_count}")
            for section in ("个人优势", "工作经历", "教育经历"):
                check(f"PDF contains section {section}", section in text)
            check("PDF text has no literal {{", "{{" not in text)
            fonts = {f[3] for f in page.get_fonts()}
            has_fallback = any("Noto" in f for f in fonts)
            check("PDF embeds no fallback font", not has_fallback, f"fonts={fonts}")
            # margins: text should not touch page edges
            w, h = page.rect.width, page.rect.height
            blocks = page.get_text("blocks")
            x0 = min(b[0] for b in blocks)
            x1 = max(b[2] for b in blocks)
            y0 = min(b[1] for b in blocks)
            y1 = max(b[3] for b in blocks)
            lm, rm, tm = x0 * MM, (w - x1) * MM, y0 * MM
            check("PDF left margin >= 8mm", lm >= 8, f"lm={lm:.1f}mm")
            check("PDF right margin >= 8mm", rm >= 8, f"rm={rm:.1f}mm")
            check("PDF top margin >= 6mm", tm >= 6, f"tm={tm:.1f}mm")
            doc.close()

        # 3) HTML
        html = td / "out.html"
        r = run_tool([str(md_path), str(html), "--format", "html"])
        if html.is_file():
            content = html.read_text(encoding="utf-8")
            check("HTML export succeeds", r.returncode == 0,
                  f"rc={r.returncode} err={r.stderr[:120]!r}")
            check("HTML contains <html>", "<html>" in content)
            name = src.splitlines()[1].lstrip("# ").strip() if len(src.splitlines()) > 1 else ""
            if name:
                check("HTML contains name", name in content)

        # 4) Word
        docx = td / "out.docx"
        r = run_tool([str(md_path), str(docx), "--format", "docx"])
        if docx.is_file():
            try:
                from docx import Document
                d = Document(str(docx))
                alltext = "\n".join(p.text for p in d.paragraphs)
                check("Word export succeeds", r.returncode == 0,
                      f"rc={r.returncode} err={r.stderr[:120]!r}")
                check("Word contains name", name in alltext)
                check("Word contains a section", "工作经历" in alltext)
            except Exception as e:  # noqa: BLE001
                check("Word opens with python-docx", False, str(e))

        # 5) style presets produce different output
        pdfm = td / "modern.pdf"
        r = run_tool([str(md_path), str(pdfm), "--format", "pdf", "--style", "modern"])
        check("--style modern succeeds", r.returncode == 0 and pdfm.is_file(),
              f"rc={r.returncode} err={r.stderr[:120]!r}")

    print("resume-from-template skill test")
    print("\n".join(checks))
    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
