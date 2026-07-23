# -*- coding: utf-8 -*-
"""
把整个项目打成 zip，方便发给同事 / 拷到 Mac。

用法：
    py -3 tools/package.py            # 默认含当前 docx + kb 数据
    py -3 tools/package.py --clean    # 不含 kb 数据（同事拿到后首次运行才会解析）
    py -3 tools/package.py --no-key   # 把 config.json 里的 api_key 清空再打包
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# 必带文件 / 目录
INCLUDE = [
    "server.py",
    "config.json",
    "README.md",
    "start.bat",
    "start.command",
    "tools/build_data.py",
    "tools/export_docx.py",
    "web/",
    # 库注册表必须带，列出有哪些库 / 各自的 system_prompt
    "kb/index.json",
]
# 数据目录（--clean 时排除）：每个库的 entries.json + images/
DATA = ["kb/"]

# 任何路径里包含这些片段都跳过（开发产物 / 缓存 / 系统文件）
SKIP_FRAG = (
    "__pycache__",
    "/.vscode/",
    "/.git/",
    "/.idea/",
    "/.DS_Store",
    "/dist/",
    "/_imported_",
    ".pyc",
    "/_test_",
    "tools/test_",
    "tools/probe.py",
    "tools/list_titles.py",
)


def should_skip(rel: str) -> bool:
    rel_norm = "/" + rel.replace("\\", "/")
    return any(frag in rel_norm for frag in SKIP_FRAG)


def collect_paths(include_data: bool) -> list[Path]:
    items: list[Path] = []
    seen: set = set()
    targets = list(INCLUDE)
    if include_data:
        targets += DATA

    def add(f: Path):
        rel = f.relative_to(ROOT).as_posix()
        if rel in seen:
            return
        seen.add(rel)
        items.append(f)

    for t in targets:
        p = ROOT / t
        if not p.exists():
            print(f"[跳过] 不存在: {t}")
            continue
        if p.is_file():
            add(p)
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(ROOT).as_posix()
                    if should_skip(rel):
                        continue
                    add(f)
    return items


def build_zip(no_key: bool, clean: bool) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = []
    if clean: suffix.append("clean")
    if no_key: suffix.append("nokey")
    name = f"wyquestion-{stamp}" + ("-" + "-".join(suffix) if suffix else "") + ".zip"
    out = DIST / name

    files = collect_paths(include_data=not clean)
    print(f"将打包 {len(files)} 个文件 -> {out.name}")

    # 处理 config.json：是否清空 key
    cfg_clean: bytes | None = None
    if no_key:
        cfg_path = ROOT / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["api_key"] = ""
        cfg_clean = (json.dumps(cfg, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    written = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            arcname = f"wyquestion/{rel}"  # 解压后是一个 wyquestion 目录

            # Mac 上的 .command / .sh 必须有执行权限才能双击运行。
            # Python zipfile 默认不写 unix 权限位，这里手动写一下。
            is_exec = rel.endswith(".command") or rel.endswith(".sh")

            if no_key and rel == "config.json":
                info = zipfile.ZipInfo(arcname)
                info.date_time = time.localtime()[:6]
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 << 16)
                zf.writestr(info, cfg_clean)
            elif is_exec:
                info = zipfile.ZipInfo.from_file(f, arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                # 高 16 位 = unix mode；0o100755 表示普通文件 + rwxr-xr-x
                info.external_attr = (0o100755 << 16)
                with open(f, "rb") as fp:
                    zf.writestr(info, fp.read())
            else:
                zf.write(f, arcname)
            written += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"完成：{out}  ({written} 个文件, {size_mb:.1f} MB)")
    if not clean and (ROOT / "kb").exists():
        print("提示：包里包含了 kb/（即当前知识库 + 图片）。要发给别人重头开始用，加 --clean。")
    if not no_key:
        print("提示：config.json 里包含了你的 api_key。给同事请用 --no-key 清掉。")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="不打包 kb/ 数据目录")
    ap.add_argument("--no-key", action="store_true", help="清空 config.json 中的 api_key")
    args = ap.parse_args()
    build_zip(no_key=args.no_key, clean=args.clean)


if __name__ == "__main__":
    main()
