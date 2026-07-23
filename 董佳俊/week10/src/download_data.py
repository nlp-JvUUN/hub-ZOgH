"""
从维基百科抓取英雄联盟相关页面的文本内容，保存为 txt 文件。
只依赖 Python 标准库，无需 pip 安装任何包。
"""
import json
import os
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 要抓取的维基百科中文页面
PAGES = [
    "英雄联盟",
    "英雄联盟角色列表",
    "英雄联盟赛事",
    "英雄联盟全球总决赛",
    "英雄联盟2018赛季全球总决赛",
    "英雄联盟2023赛季全球总决赛",
    "英雄联盟职业联赛",
    "英雄联盟韩国冠军联赛",
    "召唤师峡谷",
    "亚索",
    "李青_(英雄联盟)",
    "锐雯",
    "劫_(英雄联盟)",
    "阿狸",
    "艾希",
    "英雄联盟英雄列表",
]

USER_AGENT = "HomeworkRAG/1.0 (educational project; contact@example.com)"
API_URL = "https://zh.wikipedia.org/w/api.php"


def fetch_page(title: str) -> dict | None:
    """通过维基百科 API 获取页面的纯文本摘要。"""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    query_string = urllib.parse.urlencode(params)
    url = f"{API_URL}?{query_string}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [FAIL] 网络错误: {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_info in pages.items():
        if page_id == "-1":
            print(f"  [SKIP] 页面不存在")
            return None
        extract = page_info.get("extract", "")
        if not extract.strip():
            print(f"  [SKIP] 页面无内容")
            return None
        return {
            "title": page_info.get("title", title),
            "page_id": page_id,
            "content": extract,
            "length": len(extract),
        }

    return None


def clean_text(text: str) -> str:
    """清洗文本：去掉多余空行、参考文献标记等。"""
    # 去掉维基百科的 [1][2] 引用标记
    text = re.sub(r"\[\d+\]", "", text)
    # 去掉超长空格
    text = re.sub(r"[ \t]+", " ", text)
    # 压缩 3 个以上的空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    print("=" * 60)
    print("英雄联盟维基百科数据采集")
    print("=" * 60)

    manifest = []
    total_chars = 0

    for i, title in enumerate(PAGES, 1):
        safe_name = title.replace("/", "_").replace(" ", "_")
        print(f"[{i:02d}/{len(PAGES):02d}] {title} ... ", end="", flush=True)

        page_data = fetch_page(title)
        if page_data is None:
            continue

        page_data["content"] = clean_text(page_data["content"])
        file_name = f"{i:02d}_{safe_name}.txt"
        file_path = DATA_DIR / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {page_data['title']}\n\n")
            f.write(page_data["content"])

        total_chars += page_data["length"]
        manifest.append({
            "file": file_name,
            "title": page_data["title"],
            "page_id": page_data["page_id"],
            "length": page_data["length"],
        })
        print(f"OK ({page_data['length']} 字)")

        # 礼貌间隔，避免被限流
        time.sleep(1.0)

    # 保存 manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(manifest)}/{len(PAGES)} 页成功, 共 {total_chars:,} 字")
    print(f"文件保存在: {DATA_DIR}")


if __name__ == "__main__":
    main()
