"""word-count — 词频统计（普通函数式技能）。"""

import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]+")


def run(ctx, text: str, top_n: int = 5, **inputs):
    words = _WORD_RE.findall(text)
    count = len(words)
    chars = len(text)
    top = [w for w, _ in Counter(words).most_common(top_n)]
    return {"count": count, "chars": chars, "top_words": top}
