# -*- coding: utf-8 -*-
"""
download_bert.py — 下载 bert-base-chinese 到 pretrain_models/

网络说明：huggingface.co / hf-mirror.com 均不可达，改用 ModelScope 官方镜像：
  https://www.modelscope.cn/models/google-bert/bert-base-chinese

用法：
  python download_bert.py
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "pretrain_models" / "bert-base-chinese"

BASE = "https://www.modelscope.cn/api/v1/models/google-bert/bert-base-chinese/repo?Revision=master&FilePath="
FILES = ["config.json", "vocab.txt", "tokenizer_config.json", "model.safetensors"]


def download(name: str, max_attempts=6):
    url = BASE + name
    dst = OUT_DIR / name
    # 先取 Content-Length 作校验基准
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(head, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
    if dst.exists() and total and dst.stat().st_size == total:
        print(f"[跳过] {name} 已完整存在 ({total/1e6:.1f} MB)")
        return

    for attempt in range(1, max_attempts + 1):
        done = dst.stat().st_size if dst.exists() else 0  # 断点续传
        print(f"[下载] {name} 第 {attempt} 次尝试 (已有 {done/1e6:.1f}/{total/1e6:.1f} MB)")
        headers = {"User-Agent": "Mozilla/5.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "ab") as f:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                el = time.time() - t0
                spd = done / el / 1e6
                print(f"\r  {done/1e6:7.1f} / {total/1e6:.1f} MB  {done/total*100:5.1f}%  {spd:.1f} MB/s", end="")
        print()
        if total and done >= total:
            print(f"  {name} 完成 ({total/1e6:.1f} MB)")
            return
        print(f"  [不完整] {done/1e6:.1f} / {total/1e6:.1f} MB，重试...")
        time.sleep(2)
    raise RuntimeError(f"{name} 多次下载仍不完整")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        for attempt in range(3):
            try:
                download(name)
                break
            except Exception as e:
                print(f"\n[重试 {attempt+1}/3] {name}: {e}")
                time.sleep(2)
    print("\n全部完成:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size/1e6:8.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
