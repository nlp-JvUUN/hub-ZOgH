"""daily-report — 心跳技能：当日日志 -> MEMORY.md（Memory Flush）。"""

import sys
from datetime import date
from pathlib import Path

# skill 是独立模块，可能被任意 cwd 加载；这里兜底把 week13/ 加入 sys.path
try:
    from skillflow.journal import Journal
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from skillflow.journal import Journal


def run(ctx, **inputs):
    jdir = Path(ctx.system.get("journal_dir", Path(__file__).resolve().parent.parent.parent / "journal"))
    journal = Journal(jdir)
    summary = journal.flush()  # 规则式 Memory Flush
    return {
        "report": f"当日记忆已刷新（{date.today().isoformat()}）:\n{summary}",
        "day": date.today().isoformat(),
    }
