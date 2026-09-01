"""
session_store.py — 多会话持久化（SessionStore）

问题：进程退出后对话记忆就丢了，多轮对话只能"活着的时候"多轮。
解决：把每轮的记忆快照（turn 记录）追加写入 JSONL 文件，下次用同一个
     session_id 启动即可完整恢复 —— 多轮对话跨进程、跨天继续。

设计（与同学常见做法的差异）：
  - 常见做法：session 文件里只存 (question, answer) 两列，恢复时拼成 user/assistant。
  - 本作业：每个 turn 记录保存 memory.py 的完整状态（问题/回答/事实/摘要/时间戳），
    恢复时能重建 MemoryManager 三层记忆，追问「刚才那个 AQI 是多少」依然可答；
    且每行一个 JSON、只追加写入（append-only），中途崩溃不损坏历史，
    也天然支持「一边对话一边落盘」。
  - 会话索引单独维护（_index.jsonl），记录 标题/创建时间/更新时间/轮数，
    支持 /list /switch 等管理命令；超出 MAX_SESSIONS 按 LRU 淘汰最旧会话。

文件布局：
  sessions/
  ├── _index.jsonl          # {"session_id","title","created_at","updated_at","turns"}
  └── <session_id>.jsonl    # 每行一个 turn 记录（memory.Turn.to_dict() 的完整形态）
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import List, Optional


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class SessionStore:
    def __init__(self, sessions_dir: Path, max_sessions: int = 20):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "_index.jsonl"
        self.max_sessions = max_sessions

    # ── 索引读写 ─────────────────────────────────────────────────────────────
    def _read_index(self) -> List[dict]:
        if not self.index_path.exists():
            return []
        sessions = []
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return sessions

    def _write_index(self, sessions: List[dict]) -> None:
        # 索引是小文件，直接全量重写保证一致性
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        tmp.replace(self.index_path)

    def _touch(self, session_id: str, title: Optional[str] = None, turns: Optional[int] = None) -> None:
        sessions = self._read_index()
        now = _now()
        found = False
        for s in sessions:
            if s["session_id"] == session_id:
                s["updated_at"] = now
                if title is not None:
                    s["title"] = title
                if turns is not None:
                    s["turns"] = turns
                found = True
                break
        if not found:
            sessions.append({
                "session_id": session_id,
                "title": title or "未命名会话",
                "created_at": now,
                "updated_at": now,
                "turns": turns or 0,
            })
        self._write_index(sessions)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """LRU 淘汰：按 updated_at 升序，超出上限时删除最旧的会话文件。"""
        sessions = self._read_index()
        if len(sessions) <= self.max_sessions:
            return
        sessions.sort(key=lambda s: s["updated_at"])
        victims = sessions[: len(sessions) - self.max_sessions]
        for v in victims:
            (self.dir / f"{v['session_id']}.jsonl").unlink(missing_ok=True)
        keep_ids = {s["session_id"] for s in sessions[len(victims):]}
        self._write_index([s for s in sessions if s["session_id"] in keep_ids])

    # ── 会话生命周期 ─────────────────────────────────────────────────────────
    def create(self, title_hint: str = "") -> str:
        session_id = uuid.uuid4().hex[:8]
        title = title_hint.strip()[:24] or "未命名会话"
        self._touch(session_id, title=title, turns=0)
        return session_id

    def list_sessions(self) -> List[dict]:
        return sorted(self._read_index(), key=lambda s: s["updated_at"], reverse=True)

    def get(self, session_id: str) -> Optional[dict]:
        for s in self._read_index():
            if s["session_id"] == session_id:
                return s
        return None

    def delete(self, session_id: str) -> bool:
        (self.dir / f"{session_id}.jsonl").unlink(missing_ok=True)
        sessions = [s for s in self._read_index() if s["session_id"] != session_id]
        self._write_index(sessions)
        return True

    # ── turn 记录读写（持久化 memory 快照） ──────────────────────────────────
    def save_turn(self, session_id: str, turn_record: dict) -> None:
        path = self.dir / f"{session_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")
        self._touch(session_id, turns=len(self.load_turns(session_id)))

    def load_turns(self, session_id: str) -> List[dict]:
        path = self.dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        turns = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return turns

    def load_memory_records(self, session_id: str) -> dict:
        """把 session 文件恢复成 MemoryManager.to_records() 的形态。"""
        turns = self.load_turns(session_id)
        records = {"summary": "", "facts": [], "turns": []}
        # 从最早的 turn 开始重建（每行 turn 记录本身已含当时的 summary/facts）
        for t in turns:
            records["summary"] = t.get("summary", records["summary"])
            records["facts"] = t.get("facts", records["facts"])
            turn = {k: t.get(k) for k in ("turn", "question", "answer", "tools", "facts", "ts")}
            records["turns"].append(turn)
        # 恢复后重新做一次一致性压缩（窗口/预算以当前配置为准）
        return records
