"""
SkillFlow 自测套件 — 覆盖"渐进式"的三条主轴 + 会话/心跳/记忆。

运行方式（零第三方依赖）：
    cd week13 && python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # week13/

from skillflow.app import HarnessApp
from skillflow.discovery import Manifest, parse_frontmatter
from skillflow.engine import parse_pipeline
from skillflow.journal import Journal
from skillflow.model import Progress
from skillflow.scheduler import HEARTBEAT_SESSION, HeartbeatScheduler, parse_interval
from skillflow.session import SessionHub

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def make_app(tmp: Path, budget: int = 100) -> HarnessApp:
    return HarnessApp(
        skills_dir=SKILLS_DIR,
        state_dir=tmp / "state",
        journal_dir=tmp / "journal",
        budget=budget,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. 加载渐进
# ─────────────────────────────────────────────────────────────────────


class TestFrontmatter(unittest.TestCase):
    def test_nested_and_scalars(self):
        text = """---
name: demo
version: 1.0
weight: 5
consumes:
  text:
    type: str
    required: true
    desc: 说明文字
provides:
  count: 数量
deps: [a, b]
heartbeat: null
tags: [demo, text]
enabled: true
---
# 正文
"""
        data = parse_frontmatter(text)
        self.assertEqual(data["name"], "demo")
        self.assertEqual(data["weight"], 5)
        self.assertEqual(data["consumes"]["text"]["type"], "str")
        self.assertTrue(data["consumes"]["text"]["required"])
        self.assertEqual(data["provides"], {"count": "数量"})
        self.assertEqual(data["deps"], ["a", "b"])
        self.assertIsNone(data["heartbeat"])
        self.assertTrue(data["enabled"])

    def test_folded_string(self):
        text = """---
name: x
description: >-
  第一行
  第二行
---
"""
        data = parse_frontmatter(text)
        self.assertEqual(data["description"], "第一行 第二行")

    def test_missing_frontmatter(self):
        from skillflow.model import FrontmatterError

        with self.assertRaises(FrontmatterError):
            parse_frontmatter("# 没有 frontmatter\n正文")


class TestIncrementalScan(unittest.TestCase):
    def test_only_changed_gets_reparsed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "skills"
            (root / "alpha").mkdir(parents=True)
            (root / "alpha" / "SKILL.md").write_text(
                "---\nname: alpha\nversion: 1.0\n---\n", encoding="utf-8"
            )
            manifest = Manifest(root, Path(td) / "state")

            specs, changed1 = manifest.scan()
            self.assertEqual([s.name for s in specs], ["alpha"])
            self.assertEqual(changed1, ["alpha"])

            # 未变化：第二次扫描零变化
            _, changed2 = manifest.scan()
            self.assertEqual(changed2, [])

            # 修改 SKILL.md：只重解析 alpha
            (root / "alpha" / "SKILL.md").write_text(
                "---\nname: alpha\nversion: 2.0\n---\n", encoding="utf-8"
            )
            _, changed3 = manifest.scan()
            self.assertEqual(changed3, ["alpha"])
            self.assertEqual(manifest.get("alpha").version, "2.0")

            # 新增技能：增量发现
            (root / "beta").mkdir()
            (root / "beta" / "SKILL.md").write_text(
                "---\nname: beta\nversion: 1.0\n---\n", encoding="utf-8"
            )
            _, changed4 = manifest.scan()
            self.assertIn("beta", changed4)

            # 删除技能
            import shutil

            shutil.rmtree(root / "beta")
            _, changed5 = manifest.scan()
            self.assertIn("beta", changed5)
            self.assertIsNone(manifest.get("beta"))

    def test_resolve_order_with_deps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "skills"
            for name, deps in [("c", ["a"]), ("a", []), ("b", ["a"])]:
                d = root / name
                d.mkdir(parents=True)
                d.joinpath("SKILL.md").write_text(
                    f"---\nname: {name}\ndeps: {deps}\n---\n", encoding="utf-8"
                )
            manifest = Manifest(root, Path(td) / "state")
            manifest.scan()
            from skillflow.discovery import Registry

            reg = Registry(manifest)
            self.assertEqual(reg.resolve_order(["c"]), ["a", "c"])
            self.assertEqual(reg.resolve_order(["b", "c"]), ["a", "b", "c"])
            with self.assertRaises(ValueError):
                reg.resolve_order(["不存在"])


class TestLazyLoading(unittest.TestCase):
    def test_impl_not_loaded_until_run(self):
        with tempfile.TemporaryDirectory() as td:
            app = make_app(Path(td))
            app.scan()
            info = app.info("word-count")
            self.assertFalse(info["impl_loaded"])  # 发现 ≠ 加载（L1 与 L2 分离）
            app.run_stream("t", {"skill": "word-count", "inputs": {"text": "hi there"}})
            info = app.info("word-count")
            self.assertTrue(info["impl_loaded"])

    def test_budget_defers_heavy_skill(self):
        with tempfile.TemporaryDirectory() as td:
            app = make_app(Path(td), budget=3)  # fetch-source weight=5 -> 超预算
            events = app.run_stream("t", {"skill": "fetch-source"})
            kinds = [e.kind for e in events]
            self.assertIn("stage_defer", kinds)
            report = events[-1].payload
            self.assertEqual(report["status"], "deferred")


# ─────────────────────────────────────────────────────────────────────
# 2. 执行渐进
# ─────────────────────────────────────────────────────────────────────


class TestPipelineExecution(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.app = make_app(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_single_skill(self):
        events = self.app.run_stream("t", {"skill": "word-count", "inputs": {"text": "a b c a"}})
        report = events[-1].payload
        self.assertEqual(report["status"], "ok")
        stage = report["stages"][0]
        self.assertEqual(stage["output"]["count"], 4)

    def test_progress_events_streamed(self):
        events = self.app.run_stream("t", {"skill": "slow-progress", "inputs": {"steps": 3}})
        progress = [e for e in events if e.kind == "progress"]
        self.assertEqual(len(progress), 3)
        self.assertEqual(progress[-1].payload.percent, 100)

    def test_pipeline_contract_piping(self):
        # fetch-source(提供 text) | word-count(消费 text) | format-report(消费 count)
        events = self.app.run_stream("t", {"pipe": "fetch-source | word-count | format-report"})
        report = events[-1].payload
        self.assertEqual(report["status"], "ok")
        stages = {s["skill"]: s for s in report["stages"]}
        self.assertEqual(stages["word-count"]["status"], "ok")
        self.assertGreater(stages["word-count"]["output"]["count"], 0)
        self.assertIn("统计报告", stages["format-report"]["output"]["report"])
        # 契约注入：word-count 的 text 输入来自 fetch-source，而不是用户
        load_events = [e for e in events if e.kind == "load" and e.payload.get("stage") == "L3"]
        self.assertEqual(len(load_events), 1)

    def test_pipeline_parse(self):
        self.assertEqual(parse_pipeline("a | b | c"), ["a", "b", "c"])
        self.assertEqual(parse_pipeline("a→b"), ["a", "b"])

    def test_failure_policy_stop(self):
        events = self.app.run_stream(
            "t",
            {"pipe": "flaky-demo | word-count", "inputs": {"should_fail": True}, "config": {"on_failure": "stop"}},
        )
        report = events[-1].payload
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["stages"]), 1)  # word-count 未执行

    def test_failure_policy_skip(self):
        events = self.app.run_stream(
            "t",
            {"pipe": "flaky-demo | word-count", "inputs": {"should_fail": True}, "config": {"on_failure": "skip"}},
        )
        report = events[-1].payload
        # flaky 失败 + word-count 因缺 text 级联跳过
        statuses = [s["status"] for s in report["stages"]]
        self.assertIn("failed", statuses)
        self.assertIn("skipped", statuses)
        self.assertEqual(report["status"], "failed")  # 无 ok -> failed

    def test_failure_policy_default(self):
        events = self.app.run_stream(
            "t",
            {"pipe": "flaky-demo | format-report", "inputs": {"should_fail": True}, "config": {"on_failure": "default"}},
        )
        report = events[-1].payload
        # format-report 的 count 是必填且无默认值 -> 仍失败（无默认值可兜）
        self.assertEqual(report["status"], "failed")

    def test_l3_resource_events(self):
        events = self.app.run_stream("t", {"skill": "fetch-source"})
        loads = [e for e in events if e.kind == "load" and e.payload.get("stage") == "L3"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0].payload["resource"], "sample.txt")

    def test_hot_reload_impl(self):
        import shutil

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "skills"
            (root / "echo").mkdir(parents=True)
            (root / "echo" / "SKILL.md").write_text(
                "---\nname: echo\nversion: 1.0\nconsumes:\n  x: {type: str, required: true}\nprovides:\n  out: 输出\n---\n",
                encoding="utf-8",
            )
            (root / "echo" / "skill.py").write_text(
                "def run(ctx, x, **kw):\n    return {'out': x + '!'}\n", encoding="utf-8"
            )
            app = HarnessApp(skills_dir=root, state_dir=Path(td) / "state", journal_dir=Path(td) / "journal")
            r1 = app.run_stream("t", {"skill": "echo", "inputs": {"x": "hi"}})[-1].payload
            self.assertEqual(r1["stages"][0]["output"]["out"], "hi!")

            # 热更新实现：不重启，直接改 skill.py
            time.sleep(0.02)
            (root / "echo" / "skill.py").write_text(
                "def run(ctx, x, **kw):\n    return {'out': x.upper()}\n", encoding="utf-8"
            )
            r2 = app.run_stream("t", {"skill": "echo", "inputs": {"x": "hi"}})[-1].payload
            self.assertEqual(r2["stages"][0]["output"]["out"], "HI")


# ─────────────────────────────────────────────────────────────────────
# 3. 会话渐进（Lane 队列）
# ─────────────────────────────────────────────────────────────────────


class TestSessionLane(unittest.TestCase):
    def test_messages_processed_serially_in_order(self):
        order: list = []
        lock = threading.Lock()

        def processor(msg, publish):
            time.sleep(0.05)
            with lock:
                order.append(msg.content["tag"])
            return []

        hub = SessionHub(processor)
        for i in range(5):
            hub.submit("s1", {"tag": f"m{i}"})
        time.sleep(0.6)  # 等 Lane 消化完
        self.assertEqual(order, ["m0", "m1", "m2", "m3", "m4"])  # FIFO 严格串行

    def test_lane_error_pauses_and_resume(self):
        calls = {"n": 0}

        def processor(msg, publish):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("boom")
            return []

        hub = SessionHub(processor, max_retries=2)
        hub.submit("s1", {"x": 1})
        time.sleep(0.4)
        session = hub.get_session("s1")
        self.assertTrue(session.has_error)
        self.assertTrue(session.paused)  # 重试超限 -> Lane 暂停
        self.assertEqual(session.retry_count, 2)

        # 用户确认后恢复：当前失败消息被重试（成功），随后处理新消息
        self.assertTrue(session.resume())
        hub.submit("s1", {"x": 2})
        time.sleep(0.4)
        self.assertFalse(session.has_error)
        self.assertEqual(calls["n"], 4)  # n=3 重试成功, n=4 新消息成功


class TestHeartbeat(unittest.TestCase):
    def test_interval_parse(self):
        self.assertEqual(parse_interval("30s"), 30)
        self.assertEqual(parse_interval("5m"), 300)
        self.assertEqual(parse_interval("1h"), 3600)

    def test_run_due_now_triggers_heartbeat_skill(self):
        with tempfile.TemporaryDirectory() as td:
            app = make_app(Path(td))
            msgs = app.scheduler.run_due_now()
            self.assertTrue(any(m.content.get("skill") == "daily-report" for m in msgs))
            time.sleep(0.8)  # 等 Lane 执行完
            memory = app.journal.read_memory()
            self.assertIn("## ", memory)  # Memory Flush 已写入 MEMORY.md


class TestJournal(unittest.TestCase):
    def test_events_land_in_markdown_and_flush(self):
        with tempfile.TemporaryDirectory() as td:
            app = make_app(Path(td))
            app.run_stream("t", {"skill": "word-count", "inputs": {"text": "hello world hello"}})
            day_md = app.journal.read_day()
            self.assertIn("stage_ok", day_md)  # 每日日志（录音）
            app.flush()
            memory = app.journal.read_memory()
            self.assertIn("word-count", memory)  # MEMORY.md（纪要）


# ─────────────────────────────────────────────────────────────────────
# 5. 元技能（agent-react：ReAct 循环作为技能，mock LLM）
# ─────────────────────────────────────────────────────────────────────


class TestAgentReact(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.app = make_app(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fake_llm, question="你好"):
        events = self.app.run_stream(
            "t",
            {
                "skill": "agent-react",
                "inputs": {"question": question, "max_iterations": 6},
                "config": {"system": {"llm_client": fake_llm}},
            },
        )
        report = events[-1].payload
        return report, events

    def test_direct_final_answer(self):
        fake = lambda messages: json.dumps(
            {"action": "final_answer", "thought": "无需工具", "answer": "你好！"}
        )
        report, events = self._run(fake)
        self.assertEqual(report["status"], "ok")
        output = report["stages"][0]["output"]
        self.assertEqual(output["answer"], "你好！")
        self.assertEqual(output["iterations"], 1)
        # ReAct 轮次以 progress 事件渐进可见
        self.assertTrue(any(e.kind == "progress" for e in events))

    def test_tool_call_then_final_answer(self):
        def fake(messages):
            if any(m["role"] == "user" and "观察结果" in m["content"] for m in messages):
                return json.dumps({"action": "final_answer", "answer": "统计完成，共 3 个词"})
            return json.dumps(
                {"action": "call_tool", "tool": "word-count", "params": {"text": "a b c"}}
            )

        report, _ = self._run(fake, question="帮我统计单词数")
        output = report["stages"][0]["output"]
        self.assertEqual(output["answer"], "统计完成，共 3 个词")
        steps = output["steps"]
        call = [s for s in steps if s["action"] == "call_tool"]
        obs = [s for s in steps if s["action"] == "observation"]
        self.assertEqual(call[0]["tool"], "word-count")
        self.assertTrue(obs[0]["success"])
        self.assertIn("count", obs[0]["observation"])  # 真实执行了 word-count

    def test_tool_error_recovery(self):
        def fake(messages):
            if any(m["role"] == "user" and "观察结果" in m["content"] for m in messages):
                return json.dumps({"action": "final_answer", "answer": "该技能不存在，我直接回答"})
            return json.dumps({"action": "call_tool", "tool": "不存在的技能", "params": {}})

        report, _ = self._run(fake)
        output = report["stages"][0]["output"]
        self.assertEqual(output["answer"], "该技能不存在，我直接回答")
        obs = [s for s in output["steps"] if s["action"] == "observation"]
        self.assertFalse(obs[0]["success"])  # 失败作为观察回喂，而不是崩溃

    def test_max_iterations_guard(self):
        fake = lambda messages: json.dumps({"action": "bad_action"})
        report, _ = self._run(fake, question="x")
        output = report["stages"][0]["output"]
        self.assertIn("最大推理轮数", output["answer"])
        self.assertEqual(output["iterations"], 6)


# ─────────────────────────────────────────────────────────────────────
# 4. 网关 / 集成
# ─────────────────────────────────────────────────────────────────────


class TestHttpGateway(unittest.TestCase):
    def test_gateway_endpoints(self):
        import json
        import urllib.request

        with tempfile.TemporaryDirectory() as td:
            from skillflow.gateway import Gateway

            app = make_app(Path(td))
            gw = Gateway(app, host="127.0.0.1", port=0, watch=False)
            gw.start()
            gw.serve_in_background()
            port = gw._httpd.server_address[1]
            base = f"http://127.0.0.1:{port}"

            def get(path):
                with urllib.request.urlopen(base + path, timeout=10) as r:
                    return json.loads(r.read().decode())

            def post(path, body):
                req = urllib.request.Request(
                    base + path,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    return json.loads(r.read().decode())

            try:
                health = get("/api/health")
                self.assertEqual(health["status"], "ok")
                self.assertGreater(health["skills_count"], 0)

                skills = get("/api/skills")
                names = [s["name"] for s in skills]
                self.assertIn("word-count", names)

                sid = post("/api/sessions", {})["session_id"]
                msg = post(f"/api/sessions/{sid}/messages", {"skill": "word-count", "inputs": {"text": "a b c"}})
                self.assertIn("msg_id", msg)

                # 轮询事件直到 report
                after = 0
                report = None
                for _ in range(50):
                    data = get(f"/api/sessions/{sid}/events?after={after}")
                    after = data["next"]
                    for e in data["events"]:
                        if e["kind"] == "report":
                            report = e
                    if report:
                        break
                    time.sleep(0.1)
                self.assertIsNotNone(report)
                self.assertEqual(report["payload"]["status"], "ok")

                reloaded = post("/api/reload", {})
                self.assertIn("changed", reloaded)

                triggered = post("/api/heartbeat/run", {})
                self.assertIn("daily-report", triggered["triggered"])
            finally:
                gw.stop()

    def test_engine_generator_api(self):
        """PipelineEngine.run 本身是生成器：逐事件消费即可（渐进式执行的编程接口）。"""
        from skillflow.engine import PipelineEngine

        with tempfile.TemporaryDirectory() as td:
            app = make_app(Path(td))
            engine = PipelineEngine(app.registry, app.runtime)
            seen = [e.kind for e in engine.run("g", ["slow-progress"], {"steps": 2})]
            self.assertIn("progress", seen)
            self.assertIn("report", seen)
            self.assertEqual(seen[0], "discover")


if __name__ == "__main__":
    unittest.main(verbosity=2)
