"""
渐进式执行引擎：生成器流 + 契约式管道 + 失败降级。

「渐进式执行」在这里落地为三条机制：

  1. 流式：skill 实现是生成器，执行中 yield Progress，引擎把进度一条条
     转发成事件（CLI 打印进度、HTTP 用 SSE 推送），结果未出、过程可见。
  2. 管道：stage 的输出按 provides/consumes 契约注入下一 stage 的输入，
     数据在 stage 之间「逐级流动」，而不是等全部跑完再汇总。
  3. 降级：on_failure 策略控制失败后的行为 —— stop（整体停止）、
     skip（跳过失败 stage，下游级联跳过）、default（用声明默认值兜底）。
     无论哪种策略，已成功的 stage 结果都保留在 report.stages 里（部分结果不丢）。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .discovery import Registry
from .loader import SkillRuntime, LoadBudget
from .model import (
    BudgetExceeded,
    Event,
    ExecutionReport,
    Progress,
    SkillSpec,
    StageResult,
)


@dataclass
class StageContext:
    """
    传给 skill 运行时的上下文（"工作记忆"）。

    skill 通过它拿到：解析好的输入、规格、运行时（L3 资源懒加载）、
    当前会话 id。所有字段只读，保证 skill 之间不互相污染。
    """

    session: str
    spec: SkillSpec
    inputs: Dict[str, Any]
    runtime: SkillRuntime
    on_resource: Any = None  # 引擎注入的资源事件回调
    system: Dict[str, Any] = field(default_factory=dict)  # harness 注入的系统服务信息

    # ── L3 资源接口（用到才加载） ────────────────────────────

    def resources(self) -> List[Dict[str, Any]]:
        """资源清单（不读内容）。"""
        return self.runtime.list_resources(self.spec)

    def resource(self, name: str) -> str:
        """按需读取文本资源，并通知引擎发 load 事件。"""
        data = self.runtime.load_resource(self.spec, name)
        if self.on_resource is not None:
            self.on_resource(name, len(data))
        return data.decode("utf-8")

    def resource_bytes(self, name: str) -> bytes:
        data = self.runtime.load_resource(self.spec, name)
        if self.on_resource is not None:
            self.on_resource(name, len(data))
        return data


def parse_pipeline(expr: str) -> List[str]:
    """把 'a | b | c' / 'a,b,c' 解析成 stage 名列表。"""
    for sep in ("|", "→", "->", ","):
        if sep in expr:
            return [x.strip() for x in expr.split(sep) if x.strip()]
    return [expr.strip()]


def _normalize_output(spec: SkillSpec, value: Any) -> Dict[str, Any]:
    """把 stage 的原始输出规范成 {契约key: 值} 的字典。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    provides = spec.provides
    if len(provides) == 1:
        return {next(iter(provides)): value}
    return {"value": value}


class PipelineEngine:
    """把一组 skill（含依赖）按顺序渐进执行，产出事件流。"""

    def __init__(self, registry: Registry, runtime: Optional[SkillRuntime] = None):
        self.registry = registry
        self.runtime = runtime or SkillRuntime()

    # ── 公开入口 ──────────────────────────────────────────────

    def run(
        self,
        session: str,
        names: Iterable[str],
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Event]:
        """
        执行 skill 管道，逐个产出事件。

        config:
          on_failure: "stop" | "skip" | "default"（默认 "skip"）
          max_stages: 最大执行 stage 数（防失控）
        """
        inputs = dict(inputs or {})
        config = dict(config or {})
        policy = config.get("on_failure", "skip")
        if policy not in ("stop", "skip", "default"):
            raise ValueError(f"未知失败策略: {policy}")

        names = list(names)
        t0 = time.time()

        # ── Step 1：依赖展开 + 拓扑排序（只规划本次路径） ──
        try:
            order = self.registry.resolve_order(names)
        except (KeyError, ValueError) as e:
            yield Event("report", session, "__pipeline__", {"status": "failed", "error": str(e)})
            return

        yield Event(
            "discover",
            session,
            "__pipeline__",
            {"order": order, "budget": self.runtime.budget.to_dict()},
        )

        stages: List[StageResult] = []
        stage_outputs: List[Tuple[str, Dict[str, Any]]] = []  # 已成功的输出
        aborted = False

        # ── Step 2：逐 stage 渐进执行 ────────────────────────
        for idx, name in enumerate(order):
            if config.get("max_stages") and idx >= config["max_stages"]:
                break
            spec = self.registry.require(name)
            stage_t0 = time.time()
            result = StageResult(skill=name, status="failed")

            # 2.1 加载实现（L2；预算不足 → 推迟）
            try:
                yield Event("load", session, name, {"stage": "L2", "message": "加载实现…"})
                impl = self.runtime.get_impl(spec)
                yield Event(
                    "load",
                    session,
                    name,
                    {"stage": "L2", "message": "实现已加载", "budget": self.runtime.budget.to_dict()},
                )
            except BudgetExceeded as e:
                result.status = "deferred"
                result.error = str(e)
                stages.append(result)
                yield Event("stage_defer", session, name, {"reason": str(e)})
                continue
            except Exception as e:
                result.status = "failed"
                result.error = f"实现加载失败: {e}"
                stages.append(result)
                yield Event("stage_fail", session, name, {"error": result.error})
                if policy == "stop":
                    aborted = True
                continue

            # 2.2 组装输入：用户参数 + 上游契约注入
            stage_inputs = dict(inputs)
            for prev_name, prev_out in stage_outputs:
                for k, v in prev_out.items():
                    if k in spec.consumes and k not in stage_inputs:
                        stage_inputs[k] = v

            # 2.3 校验必填输入（失败时按策略处理）
            missing = [k for k in spec.required_inputs() if k not in stage_inputs]
            # 管道中任一前置 stage 未成功（失败/跳过/推迟）都视为"上游断裂"
            upstream_broken = any(s.status != "ok" for s in stages)
            if missing:
                reason = f"缺少必填输入: {', '.join(missing)}"
                if upstream_broken and policy == "skip":
                    result.status = "skipped"
                    result.error = f"上游未产出，级联跳过（{reason}）"
                elif policy == "default":
                    for k in missing:
                        stage_inputs[k] = spec.default_for(k)
                    if any(stage_inputs.get(k) is None for k in missing):
                        result.status = "failed"
                        result.error = reason
                    else:
                        result.status = "ok"  # 先置 ok，下面执行失败会再覆盖
                        result.error = ""
                else:
                    result.status = "failed"
                    result.error = reason
                if result.status != "ok":
                    stages.append(result)
                    yield Event(
                        "stage_skip" if result.status == "skipped" else "stage_fail",
                        session,
                        name,
                        {"reason": result.error},
                    )
                    if policy == "stop" and result.status == "failed":
                        aborted = True
                    continue

            # 2.4 执行（生成器流式 / 普通函数）
            event_sink: List[Event] = []

            def resource_event(rn: str, size: int, sink=event_sink):
                # skill 通过 ctx.resource() 读 L3 资源时，引擎同步发 load 事件
                sink.append(Event("load", session, name, {"stage": "L3", "resource": rn, "size": size}))

            ctx = StageContext(
                session=session,
                spec=spec,
                inputs=stage_inputs,
                runtime=self.runtime,
                on_resource=resource_event,
                system=config.get("system", {}),
            )

            try:
                output, gen_events = self._invoke(impl, ctx)
                event_sink.extend(gen_events)
                result.status = "ok"
                result.output = output
            except Exception as e:
                result.status = "failed"
                result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            finally:
                result.duration_ms = (time.time() - stage_t0) * 1000

            for ev in event_sink:
                yield ev
            stages.append(result)

            if result.status == "ok":
                norm = _normalize_output(spec, result.output)
                stage_outputs.append((name, norm))
                yield Event("stage_ok", session, name, {"output": norm, "duration_ms": result.duration_ms})
            else:
                yield Event("stage_fail", session, name, {"error": result.error})
                if policy == "stop":
                    aborted = True

            if aborted:
                break

        # ── Step 3：汇总报告 ─────────────────────────────────
        ok_count = sum(1 for s in stages if s.status == "ok")
        deferred_count = sum(1 for s in stages if s.status == "deferred")
        if aborted:
            status = "failed"
            message = "按 stop 策略中止"
        elif stages and deferred_count == len(stages):
            status = "deferred"
            message = "全部 stage 因加载预算不足被推迟"
        elif ok_count == len(stages) and stages:
            status = "ok"
            message = "全部 stage 成功"
        elif ok_count > 0:
            status = "partial"
            message = "部分 stage 成功（其余失败/跳过/推迟）"
        elif stages:
            status = "failed"
            message = "无 stage 成功"
        else:
            status = "failed"
            message = "空管道"

        report = ExecutionReport(
            session=session,
            status=status,
            stages=stages,
            events=[],
            duration_ms=(time.time() - t0) * 1000,
            message=message,
        )
        yield Event("report", session, "__pipeline__", report.to_dict())

    # ── 内部 ─────────────────────────────────────────────────

    @staticmethod
    def _invoke(impl, ctx: StageContext) -> Tuple[Any, List[Event]]:
        """
        调用 skill 实现，适配三种写法：
          生成器（yield Progress / yield 最终值 / return 最终值）、普通函数、Skill 类。
        """
        events: List[Event] = []
        ret = impl(ctx, **ctx.inputs)

        if hasattr(ret, "__next__"):  # 生成器 → 流式
            final = None
            while True:
                try:
                    item = next(ret)
                except StopIteration as stop:
                    # 生成器 return 值（必须手动 next 才能拿到 StopIteration.value）
                    if stop.value is not None:
                        final = stop.value
                    break
                if isinstance(item, Progress):
                    events.append(Event("progress", ctx.session, ctx.spec.name, item))
                else:
                    final = item  # 提前 yield 的最终输出
            return final, events

        return ret, events

    def run_collect(
        self,
        session: str,
        names: Iterable[str],
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        """便捷版：跑完整个管道，返回汇总报告（事件也保留在报告里）。"""
        events = list(self.run(session, names, inputs, config))
        last = events[-1]
        payload = last.payload
        report = ExecutionReport(
            session=session,
            status=payload.get("status", "failed"),
            stages=payload.get("stages", []),
            events=events,
            duration_ms=payload.get("duration_ms", 0),
            message=payload.get("message", ""),
        )
        return report
