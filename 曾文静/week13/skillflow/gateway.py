"""
HTTP 网关 — 把 harness 暴露成服务（课件 Fat Gateway 的落地形态之一）。

路由一览：
    GET  /api/health                     健康检查
    GET  /api/skills                     技能清单（L1 元数据，不加载实现）
    GET  /api/skills/<name>              单个技能详情（含 L3 资源清单）
    POST /api/sessions                   创建会话 -> {"session_id"}
    GET  /api/sessions                   会话列表（Lane 状态）
    POST /api/sessions/<sid>/messages    投递消息（skill/pipe）-> 202 + msg_id
    GET  /api/sessions/<sid>/events      增量拉取事件（?after=N，普通轮询）
    GET  /api/sessions/<sid>/stream      SSE 实时事件流（渐进式执行的过程可见）
    POST /api/reload                     增量重扫（热更新入口）
    POST /api/heartbeat/run              立即触发全部心跳技能
    POST /api/flush                      当日日志 -> MEMORY.md（Memory Flush）
    GET  /api/journal                    当日 Markdown 日志
    GET  /api/memory                     MEMORY.md 内容

零第三方依赖：http.server + json + threading。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .app import HarnessApp

API_PREFIX = "/api"


def _json(data: Any, code: int = 200, ctype: str = "application/json; charset=utf-8") -> bytes:
    return (json.dumps(data, ensure_ascii=False).encode("utf-8"), code, ctype)


class GatewayHandler(BaseHTTPRequestHandler):
    """一个连接一个线程；所有状态都在 self.server.app 上。"""

    # 关闭默认的日志噪音，改为精简一行
    def log_message(self, fmt, *args):
        pass

    @property
    def app(self) -> HarnessApp:
        return self.server.app  # type: ignore[attr-defined]

    # ── 路由 ─────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path
        q = urlparse(self.path).query
        params = dict(pair.split("=", 1) for pair in q.split("&") if "=" in pair)

        if path == API_PREFIX + "/health":
            return self._send(*_json({"status": "ok", **self.app.status()}))
        if path == API_PREFIX + "/skills":
            return self._send(*_json([s.to_dict() for s in self.app.registry.list_all()]))
        if path.startswith(API_PREFIX + "/skills/"):
            name = path[len(API_PREFIX + "/skills/") :]
            try:
                return self._send(*_json(self.app.info(name)))
            except KeyError as e:
                return self._send(*_json({"error": str(e)}, 404))
        if path == API_PREFIX + "/sessions":
            return self._send(*_json(self.app.hub.list_sessions()))
        if path.startswith(API_PREFIX + "/sessions/") and path.endswith("/events"):
            sid = path[len(API_PREFIX + "/sessions/") : -len("/events")]
            after = int(params.get("after", "0"))
            timeout = float(params.get("timeout", "15"))
            events, next_idx = self._poll_events(sid, after, timeout)
            return self._send(*_json({"events": events, "next": next_idx}))
        if path.startswith(API_PREFIX + "/sessions/") and path.endswith("/stream"):
            sid = path[len(API_PREFIX + "/sessions/") : -len("/stream")]
            return self._stream_sse(sid, int(params.get("after", "0")))
        if path == API_PREFIX + "/journal":
            day = params.get("day")
            from datetime import date

            d = date.fromisoformat(day) if day else date.today()
            return self._send(*_json({"day": d.isoformat(), "content": self.app.journal.read_day(d)}))
        if path == API_PREFIX + "/memory":
            return self._send(*_json({"content": self.app.journal.read_memory()}))
        return self._send(*_json({"error": "not found"}, 404))

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json()

        if path == API_PREFIX + "/sessions":
            sid = body.get("session_id") or f"s-{uuid.uuid4().hex[:8]}"
            self.app.hub.get_session(sid)
            return self._send(*_json({"session_id": sid}, 201))
        if path.startswith(API_PREFIX + "/sessions/") and path.endswith("/messages"):
            sid = path[len(API_PREFIX + "/sessions/") : -len("/messages")]
            msg = self.app.hub.submit(sid, body)
            return self._send(*_json({"msg_id": msg.msg_id, "queue_depth": msg.metadata.get("depth", 0)}, 202))
        if path == API_PREFIX + "/reload":
            return self._send(*_json(self.app.scan(force=True)))
        if path == API_PREFIX + "/heartbeat/run":
            msgs = self.app.scheduler.run_due_now()
            return self._send(*_json({"triggered": [m.content.get("skill") for m in msgs]}))
        if path == API_PREFIX + "/flush":
            summary = self.app.flush(body.get("day"))
            return self._send(*_json({"summary": summary}))
        return self._send(*_json({"error": "not found"}, 404))

    # ── 实现细节 ─────────────────────────────────────────────

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _poll_events(self, sid: str, after: int, timeout: float):
        """长轮询：有新事件立即返回，否则等到超时。"""
        deadline = time.time() + min(timeout, 60)
        while time.time() < deadline:
            events = self.app.hub.events(sid, after=after)
            if events:
                return events, after + len(events)
            time.sleep(0.2)
        return [], self.app.hub.event_count()

    def _stream_sse(self, sid: str, after: int):
        """SSE：把该会话的事件实时推给浏览器/客户端（渐进式输出的 HTTP 形态）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        idx = after
        try:
            while True:
                events = self.app.hub.events(sid, after=idx)
                for e in events:
                    self.wfile.write(f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    idx += 1
                if not events:
                    time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, data: bytes, code: int, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Gateway:
    """网关服务：管理 app、端口、watch 与心跳后台线程。"""

    def __init__(self, app: HarnessApp, host: str = "127.0.0.1", port: int = 8620, watch: bool = True):
        self.app = app
        self.host = host
        self.port = port
        self.watch = watch
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._serving = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self) -> "Gateway":
        self._httpd = ThreadingHTTPServer((self.host, self.port), GatewayHandler)
        self._httpd.app = self.app  # type: ignore[attr-defined]
        if self.watch:
            self.app.start_watch()
        self._heartbeat_thread = threading.Thread(
            target=self.app.scheduler.run_forever, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_in_background(self) -> threading.Thread:
        """测试/嵌入场景：在后台线程里接受连接。"""
        t = threading.Thread(target=self._serve_loop, daemon=True, name="httpd")
        t.start()
        return t

    def _serve_loop(self):
        self._serving = True
        try:
            self._httpd.serve_forever()
        finally:
            self._serving = False

    def serve_forever(self):
        """CLI 场景：主线程阻塞服务，Ctrl+C 退出。"""
        print(f"SkillFlow 网关已启动: {self.url}  (Ctrl+C 退出)")
        try:
            self._serve_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.app.stop_watch()
        self.app.scheduler.stop()
        if self._httpd:
            if self._serving:  # 仅当 serve 循环在跑时才 shutdown，否则会永久阻塞
                self._httpd.shutdown()
            self._httpd.server_close()
