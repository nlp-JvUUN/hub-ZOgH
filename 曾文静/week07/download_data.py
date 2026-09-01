# -*- coding: utf-8 -*-
"""
week07/download_data.py
=======================
cluener2020 数据下载与解析
- 优先从 CLUE 官方 Google Storage 下载 zip (~1MB)
- 失败时回退到 GitHub raw 逐个下载 train/dev/test
- 解析为统一的 records 格式: [{"text": ..., "label": {...}}, ...]
  保存到 data/cluener/{train,validation,test}.json

用法:
  python download_data.py
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

CLUE_ZIP_URL = "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip"
GITHUB_SPLIT_URL = "https://raw.githubusercontent.com/CLUEbenchmark/CLUENER2020/master/data/{split}.json"

# 下载后按此映射重命名: 原文件名 -> 标准 split 名
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def _http_get(url: str, timeout: int = 60) -> bytes:
    """下载 URL 并返回字节内容"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_from_clue(data_dir: Path) -> bool:
    """从 CLUE Google Storage 下载 zip 并解析"""
    print(f"[1/2] 尝试 CLUE 官方源: {CLUE_ZIP_URL}")
    try:
        raw = _http_get(CLUE_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            # zip 内文件名形如 train.json / dev.json / test.json
            found = {}
            for n in names:
                stem = Path(n).stem          # 'train.json' -> 'train'
                if stem in SPLIT_MAP:
                    found[SPLIT_MAP[stem]] = n
            if not found:
                print(f"    zip 内未找到预期文件, 实际内容: {names}")
                return False
            for out_name, zip_name in found.items():
                data = zf.read(zip_name).decode("utf-8")
                records = parse_records(data)
                save_records(data_dir, out_name, records)
        return True
    except Exception as e:
        print(f"    CLUE 源失败: {type(e).__name__}: {e}，回退 GitHub 源")
        return False


def download_from_github(data_dir: Path) -> bool:
    """回退: 从 CLUEbenchmark/CLUENER2020 仓库逐个下载"""
    print("[2/2] 尝试 GitHub 源: CLUEbenchmark/CLUENER2020/data/")
    ok = True
    for src_name, out_name in SPLIT_MAP.items():
        url = GITHUB_SPLIT_URL.format(split=src_name)
        try:
            records = parse_records(_http_get(url).decode("utf-8"))
            save_records(data_dir, out_name, records)
            print(f"    {src_name} -> {out_name}: {len(records)} 条")
        except Exception as e:
            print(f"    {src_name} 下载失败: {type(e).__name__}: {e}")
            ok = False
    return ok


def parse_records(text: str) -> list:
    """每行一个 JSON 对象: {"text":..., "label": {...}}"""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append({"text": obj.get("text", ""), "label": obj.get("label", {})})
    return records


def save_records(data_dir: Path, split: str, records: list):
    """保存为缩进 JSON, 便于人工检查"""
    out = data_dir / f"{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"    {split}.json: {len(records)} 条 -> {out}")


def main():
    parser = argparse.ArgumentParser(description="cluener2020 数据下载与解析")
    parser.add_argument("--data_dir", type=str, default="data/cluener",
                        help="数据保存目录(默认 data/cluener)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not download_from_clue(data_dir):
        download_from_github(data_dir)

    print(f"\n完成! 数据位于 {data_dir}/, 包含 train/validation/test 三个文件")


if __name__ == "__main__":
    main()
