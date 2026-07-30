"""fetch-source — L3 资源按需读取（数据入口，管道第一段）。"""

import time

from skillflow.model import Progress


def run(ctx, file: str = "sample.txt", **inputs):
    yield Progress(done=1, total=3, message=f"检查资源清单（{file}）…")
    files = ctx.resources()

    yield Progress(done=2, total=3, message=f"按需读取 resources/{file}…")
    text = ctx.resource(file)  # 触发引擎的 L3 load 事件

    time.sleep(0.3)  # 模拟网络/磁盘开销
    yield Progress(done=3, total=3, message="读取完成")

    return {
        "text": text,
        "source": f"{file}（{len(text)} 字符，清单共 {len(files)} 个资源）",
    }
