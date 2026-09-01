"""format-report — 报告排版（管道终端技能）。"""


def run(ctx, count: int, chars: int = 0, top_words=None, **inputs):
    top_words = top_words or []
    lines = [
        "📊 文本统计报告",
        f"  - 单词总数: {count}",
        f"  - 字符总数: {chars}",
        f"  - 高频词: {'、'.join(top_words) if top_words else '（无）'}",
    ]
    return {"report": "\n".join(lines)}
