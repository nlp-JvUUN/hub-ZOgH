# -*- coding: utf-8 -*-
"""
向量索引模块（仅用 Python 标准库 + urllib）

功能：
- 调 embedding API（OpenAI 兼容协议，比如硅基流动的 BAAI/bge-m3）
- 向量持久化到 kb/<kbid>/vectors.json
- 内容指纹（md5）判断是否需要重新嵌入，避免重复花钱
- 余弦相似度计算（纯 Python 实现，速度对千级条目够用）

向量文件结构：
{
  "model": "BAAI/bge-m3",
  "dim": 1024,
  "items": {
    "<anchor>": {"hash": "<md5>", "vec": [0.01, -0.45, ...]}
  }
}
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import ssl
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


# -------- 工具：把 entry 抽成纯文本（embedding 输入）--------
class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "tr", "td", "th"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def entry_to_text(entry: dict, max_chars: int = 2000) -> str:
    """把 entry 拼成 embedding 用的文本：标题 + 正文（截断）。"""
    title = entry.get("title", "")
    if entry.get("html"):
        s = _Stripper()
        s.feed(entry["html"])
        body = "".join(s.parts)
    else:
        parts = []
        for b in entry.get("blocks", []) or []:
            t = b.get("type")
            if t in ("p", "h4"):
                parts.append(b.get("text", ""))
            elif t == "table":
                for row in b.get("rows", []) or []:
                    parts.append(" | ".join(row))
        body = "\n".join(parts)
    body = re.sub(r"\n{2,}", "\n", body).strip()
    text = f"{title}\n{body}"
    return text[:max_chars]


def entry_hash(entry: dict) -> str:
    """根据标题 + 正文（剔除 id 等动态字段）生成稳定指纹。"""
    text = entry_to_text(entry, max_chars=10**9)
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


# -------- 调 embedding API --------
def call_embed_api(texts: list[str], cfg: dict) -> list[list[float]] | None:
    """调用 OpenAI 兼容的 /embeddings 接口，返回每条 text 对应的向量。
    出错返回 None，调用方需要 fallback。
    """
    base_url = (cfg.get("embed_base_url") or "").rstrip("/")
    api_key = cfg.get("embed_api_key") or ""
    model = cfg.get("embed_model") or "BAAI/bge-m3"

    if not base_url or not api_key:
        return None

    payload = json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "wyquestion-local/1.0",
        },
        method="POST",
    )
    verify_ssl = cfg.get("verify_ssl", True)
    if not verify_ssl:
        ctx = ssl._create_unverified_context()
    else:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data") or []
        # 按 index 排序，确保和入参一一对应
        items.sort(key=lambda x: x.get("index", 0))
        return [it["embedding"] for it in items if "embedding" in it]
    except Exception as ex:
        print(f"[embed] 调用 embedding API 失败: {ex}")
        return None


# -------- 向量索引存储 --------
def vectors_path(kb_dir: Path) -> Path:
    return kb_dir / "vectors.json"


def load_vectors(kb_dir: Path) -> dict:
    p = vectors_path(kb_dir)
    if not p.exists():
        return {"model": "", "dim": 0, "items": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"model": "", "dim": 0, "items": {}}


def save_vectors(kb_dir: Path, data: dict) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    vectors_path(kb_dir).write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


# -------- 主流程：同步索引 --------
def reindex_kb(kb_dir: Path, entries: list[dict], cfg: dict, batch: int = 32) -> dict:
    """让向量索引和 entries 保持一致。
    返回统计信息：{added, updated, removed, skipped, ok}
    """
    stats = {"added": 0, "updated": 0, "removed": 0, "skipped": 0, "failed": 0, "ok": True, "model": ""}
    if not cfg.get("embed_enabled", True):
        stats["ok"] = False
        stats["error"] = "embed_enabled=false"
        return stats

    model = cfg.get("embed_model") or "BAAI/bge-m3"
    store = load_vectors(kb_dir)
    # 模型换了 → 整体重建
    if store.get("model") and store.get("model") != model:
        store = {"model": model, "dim": 0, "items": {}}

    # 决定要重新嵌入的 entries
    items: dict = store.get("items") or {}
    current_anchors = set()
    to_embed: list[tuple[str, str]] = []  # [(anchor, text)]

    for e in entries:
        anchor = e.get("anchor")
        if not anchor:
            continue
        current_anchors.add(anchor)
        h = entry_hash(e)
        old = items.get(anchor)
        if old and old.get("hash") == h and old.get("vec"):
            stats["skipped"] += 1
            continue
        text = entry_to_text(e)
        if not text.strip():
            stats["skipped"] += 1
            continue
        to_embed.append((anchor, text))
        items[anchor] = {"hash": h, "vec": []}  # 占位

    # 删掉已不存在的
    removed_keys = [k for k in items.keys() if k not in current_anchors]
    for k in removed_keys:
        items.pop(k, None)
    stats["removed"] = len(removed_keys)

    # 分批调 API
    for i in range(0, len(to_embed), batch):
        chunk = to_embed[i:i + batch]
        texts = [t for _, t in chunk]
        vecs = call_embed_api(texts, cfg)
        if not vecs or len(vecs) != len(chunk):
            # 失败：回滚这批占位（避免下次又用一份空向量）
            for anchor, _ in chunk:
                # 该 anchor 在 items 里现在是占位的空 vec，删掉，下次再试
                v = items.get(anchor)
                if v and not v.get("vec"):
                    items.pop(anchor, None)
            stats["failed"] += len(chunk)
            stats["ok"] = False
            stats["error"] = "embedding API 失败（已部分回滚）"
            break
        for (anchor, _), vec in zip(chunk, vecs):
            old = items.get(anchor)
            old["vec"] = vec
            if not store.get("dim"):
                store["dim"] = len(vec)
            # 增 vs 改
            if old.get("hash") and old.get("vec"):
                # 这里没法精确知道"原本就有 vs 第一次添加"
                pass
        # 简化：暂用 to_embed 长度区分
        stats["added"] += len(chunk)

    store["model"] = model
    store["items"] = items
    save_vectors(kb_dir, store)
    stats["model"] = model
    return stats


def embed_query(query: str, cfg: dict) -> list[float] | None:
    """对单条查询做 embedding。"""
    if not query or not cfg.get("embed_enabled", True):
        return None
    vecs = call_embed_api([query], cfg)
    if not vecs:
        return None
    return vecs[0]


# -------- 向量检索 --------
def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return s / (math.sqrt(na) * math.sqrt(nb))


def vector_search(kb_dir: Path, query_vec: list[float], top_k: int = 10) -> list[tuple[float, str]]:
    """返回 [(score, anchor)] 按余弦相似度降序。"""
    store = load_vectors(kb_dir)
    items = store.get("items") or {}
    if not items:
        return []
    scored = []
    for anchor, v in items.items():
        vec = v.get("vec")
        if not vec:
            continue
        scored.append((cosine(query_vec, vec), anchor))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# -------- Reranker（重排序）--------
def call_rerank_api(query: str, documents: list[str], cfg: dict,
                    top_n: int = 5) -> list[tuple[float, int]] | None:
    """调用硅基流动 /rerank 接口，返回 [(relevance_score, doc_index)] 降序。
    出错返回 None，调用方退化为不用 rerank。
    """
    base_url = (cfg.get("embed_base_url") or "").rstrip("/")
    api_key = cfg.get("embed_api_key") or ""
    model = cfg.get("rerank_model") or "BAAI/bge-reranker-v2-m3"

    if not base_url or not api_key or not documents:
        return None

    payload = json.dumps({
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        base_url + "/rerank",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "wyquestion-local/1.0",
        },
        method="POST",
    )
    verify_ssl = cfg.get("verify_ssl", True)
    if not verify_ssl:
        ctx = ssl._create_unverified_context()
    else:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results") or []
        return [(r["relevance_score"], r["index"]) for r in results
                if "relevance_score" in r and "index" in r]
    except Exception as ex:
        print(f"[rerank] 调用 rerank API 失败: {ex}")
        return None
