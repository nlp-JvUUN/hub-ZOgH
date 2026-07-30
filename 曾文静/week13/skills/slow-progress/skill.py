"""slow-progress — 进度流演示（生成器技能）。"""

import time

from skillflow.model import Progress


def run(ctx, steps: int = 5, label: str = "任务", **inputs):
    steps = max(1, int(steps))
    for i in range(1, steps + 1):
        time.sleep(0.15)
        yield Progress(done=i, total=steps, message=f"{label} 第 {i}/{steps} 步完成")
    return {"summary": f"{label} 共 {steps} 步全部完成"}
