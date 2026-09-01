"""flaky-demo — 失败注入演示。"""


def run(ctx, should_fail: bool = False, fallback: str = "默认值兜底报告", **inputs):
    if should_fail:
        raise RuntimeError("演示性失败：flaky-demo 按请求失败了")
    return {"status": "ok", "report": "flaky-demo 正常完成"}
