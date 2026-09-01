"""
agent_loop.py — 把"天气查询工具调用"改造为"循环调用"（Agent Loop）

作业核心（对比课件 mode_function_call 的单轮闭环）：
  单轮：User → LLM → tool_call → 执行 → 回填 → LLM 最终回答（固定 2 次 LLM 调用）
  循环：User → [LLM → tool_call → 执行 → 回填] × N → LLM 最终回答（N 由模型决定）

  LLM 每轮自主决定三件事：调哪个工具、调几个（支持一轮并行多个）、
  是继续调还是给出最终答案。宿主只负责"执行 + 回填 + 兜底"。

本实现与常见作业的差异点（为什么这样设计）：
  1. 工具拆成 4 个原子工具（geocode / get_current_weather / get_daily_forecast /
     get_air_quality），比"2 工具"产生更多调用链形态，且同轮可并行多个工具；
  2. 循环的价值不止"链式执行"，更在于失败自愈：工具返回 [ERROR]（城市不存在、
     坐标越界、网络失败）会作为 tool 结果回填给模型，模型在下一轮修正参数重试
     或如实告知用户 —— 单轮闭环遇到工具报错只能把错误原文丢给用户；
  3. 三层终止保护：模型主动停（不再输出 tool_calls）、最大轮数 MAX_STEPS、
     死循环检测（连续多轮调用同一工具同一参数 → 强制终止）；
  4. 可观测性：逐轮打印轨迹 + --transcript 导出 JSONL 完整记录（含 token 统计），
     方便核对循环行为；
  5. --mock 模拟驱动：内置脚本化"决策器"模拟 LLM 的 tool_call 决策，
     没有 API Key 也能完整跑通循环机制（含失败自愈场景），
     真实验证循环逻辑后再用真实模型跑。

使用方式：
  # 真实模型（需 DEEPSEEK_API_KEY，参考课件 PROVIDERS 配置）
  python agent_loop.py -q "宁德今天的天气怎么样？"
  python agent_loop.py --demo

  # 模拟驱动（无需 API Key，离线演示循环机制）
  python agent_loop.py --demo --mock
  python agent_loop.py -q "北京的天气怎么样（模拟第一轮手滑传错坐标）" --mock --transcript out.jsonl

依赖：pip install openai httpx
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# 让 weather_tools 可被 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent))

from weather_tools import (  # noqa: E402
    TOOLS, geocode, get_current_weather, get_daily_forecast,
    get_air_quality, get_comfort_index,
)

# ── LLM 配置（与课件 mode_function_call 同一套）──────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# ── 工具 Schema（手写 JSON Schema，description 写清"何时单独用/何时链式"）──────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "把中文城市名解析成经纬度（地理编码）。返回 JSON 字符串，"
                "包含 name/country/admin1/latitude/longitude 字段。"
                "用法：用户只问某城市的经纬度/坐标时，单独调用本工具即可回答；"
                "用户问某城市天气/预报/空气质量时，必须先调用本工具拿到经纬度，"
                "再把结果里的 latitude/longitude 传给其它工具（链式调用）。"
                "找不到城市时返回 [ERROR][NOT_FOUND]，请根据提示换写法重试一次，"
                "仍失败则如实告诉用户，不要编造坐标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "按经纬度查询当前天气（天气状况/温度/湿度/风速）。"
                "用户已直接给出坐标时直接调用；坐标来自 geocode 时链式调用。"
                "注意参数顺序：latitude 是纬度（南北），longitude 是经度（东西），别写反。"
                "坐标越界会返回 [ERROR][PARAM]，请修正参数后重试。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，范围 [-90, 90]，如 26.66"},
                    "longitude": {"type": "number", "description": "经度，范围 [-180, 180]，如 119.52"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_forecast",
            "description": (
                "按经纬度查询未来 days 天（1~7）逐日预报。"
                "仅当用户问的是未来多天的天气/预报时使用；只问今天用 get_current_weather。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，范围 [-90, 90]"},
                    "longitude": {"type": "number", "description": "经度，范围 [-180, 180]"},
                    "days": {"type": "integer", "description": "预报天数 1~7，默认 3"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": (
                "按经纬度查询空气质量（US AQI 指数/等级/PM2.5/PM10/O₃/NO₂）。"
                "用户问空气质量/污染/AQI 时使用；可与 get_current_weather 在同一轮并行调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，范围 [-90, 90]"},
                    "longitude": {"type": "number", "description": "经度，范围 [-180, 180]"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_comfort_index",
            "description": (
                "下游衍生工具：根据温度/湿度/风速估算体感温度与舒适度等级。"
                "不要直接调用本工具，除非你先调用了 get_current_weather 并从中提取参数："
                "从 get_current_weather 返回文本中提取 '温度：X°C'、'相对湿度：Y%'、"
                "'风速：Z km/h' 三行数值，分别作为 temperature/humidity/wind_speed 传入。"
                "用户问'体感/舒适度/热不热/冷不冷'时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number", "description": "当前温度（°C），从 get_current_weather 返回文本中提取"},
                    "humidity": {"type": "number", "description": "相对湿度（%），从 get_current_weather 返回文本中提取"},
                    "wind_speed": {"type": "number", "description": "风速（km/h），从 get_current_weather 返回文本中提取"},
                },
                "required": ["temperature", "humidity", "wind_speed"],
            },
        },
    },
]

# 工具名 → 后端函数（业务逻辑在 weather_tools.py，本文件只做派发）
TOOL_DISPATCH = {
    "geocode": geocode,
    "get_current_weather": get_current_weather,
    "get_daily_forecast": get_daily_forecast,
    "get_air_quality": get_air_quality,
    "get_comfort_index": get_comfort_index,
}

SYSTEM_PROMPT = (
    "你是一名智能天气助手，拥有 5 个专业工具：geocode（城市名→经纬度）、"
    "get_current_weather（当前天气）、get_daily_forecast（未来多天预报）、"
    "get_air_quality（空气质量）、get_comfort_index（体感舒适度）。\n\n"
    "工作流程：\n"
    "- 用户只问城市经纬度 → 只调 geocode；\n"
    "- 用户问城市天气/预报/空气质量 → 先 geocode 拿经纬度，再按需调用对应工具；\n"
    "- 用户直接给出经纬度 → 跳过 geocode，直接调用天气/空气质量工具；\n"
    "- 用户问体感/舒适度 → geocode → get_current_weather → 从返回文本中提取"
    "温度/湿度/风速 → get_comfort_index（下游衍生链式调用）；\n"
    "- 需要同时知道天气和空气质量时，可以在同一轮并行调用两个工具。\n\n"
    "错误处理：\n"
    "- 工具返回 [ERROR][NOT_FOUND]/[ERROR][PARAM] 时，先根据错误提示修正参数重试一次；\n"
    "- 重试仍失败，如实告知用户原因，绝不编造数据；\n"
    "- 严格依据工具返回的真实数据回答，不要猜测。"
)


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY（或使用 --mock 模拟驱动）", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 模拟决策器（Mock LLM）：无 API Key 也能完整演示循环机制 ───────────────────

class MockPlanner:
    """
    用规则脚本模拟 LLM 的"决策"：每次调用返回 {content, tool_calls}，
    与真实模型响应的规范化结构完全一致，因此 AgentLoop.run 无需区分真假驱动。

    演示覆盖：
      - 链式：geocode → get_current_weather / get_daily_forecast / get_air_quality
      - 下游衍生链式：geocode → get_current_weather → 提取温湿度风速 → get_comfort_index
      - 并行：一轮输出多个 tool_call（天气 + 空气质量）
      - 失败自愈：故意第一轮传错坐标 → 收到 [ERROR][PARAM] → 第二轮修正重试
      - 诚实拒答：城市不存在 → geocode 返回 [ERROR][NOT_FOUND] → 如实告知用户
    """

    def __init__(self, tool_names, verbose: bool = True):
        self.tool_names = tool_names
        self.verbose = verbose
        self.city_db = {
            "宁德": {"name": "宁德", "country": "中国", "admin1": "福建省",
                     "latitude": 26.66, "longitude": 119.52},
            "北京": {"name": "北京", "country": "中国", "admin1": "北京市",
                     "latitude": 39.90, "longitude": 116.40},
        }
        self._intent = None      # current / forecast / air / both / coords / comfort
        self._city = None
        self._coords = None      # 已拿到的经纬度
        self._geo_done = False
        self._typo_pending = False   # 自愈演示开关
        self._call_seq = 0

    # ---- 与真实 LLM 同构的入口 ----
    def decide(self, messages):
        last = messages[-1]
        if last["role"] == "user":
            return self._plan(last["content"])
        return self._react(last)

    def _tool_call(self, name, args):
        self._call_seq += 1
        return {
            "id": f"call_mock_{self._call_seq}",
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    def _plan(self, question: str):
        # 每题独立：重置上一题遗留的决策状态
        self._intent = None
        self._city = None
        self._coords = None
        self._geo_done = False
        self._typo_pending = False
        self._city = next((c for c in self.city_db if c in question), None)
        has_weather, has_air = "天气" in question, "空气" in question
        if any(k in question for k in ("舒适", "体感")):
            self._intent = "comfort"       # 下游衍生链式：天气 → 提取参数 → 舒适度
        elif has_weather and has_air:
            self._intent = "both"          # 天气 + 空气质量 → 并行两个工具
        elif has_air:
            self._intent = "air"
        elif any(k in question for k in ("预报", "未来", "几天")):
            self._intent = "forecast"
        elif has_weather:
            self._intent = "current"
        elif any(k in question for k in ("经纬度", "坐标")):
            self._intent = "coords"
        else:
            self._intent = "current"
        self._typo_pending = "手滑" in question  # 自愈演示问题自带标记

        # 用户直接给了坐标：跳过 geocode
        m = re.search(r"经度\s*([-\d.]+).*?纬度\s*([-\d.]+)", question) or \
            re.search(r"纬度\s*([-\d.]+).*?经度\s*([-\d.]+)", question)
        if m:
            lon, lat = float(m.group(1)), float(m.group(2))
            self._coords = (lat, lon)
            self._geo_done = True
            return self._next_weather_call()

        if self._city:
            return {"content": None, "tool_calls": [self._tool_call("geocode", {"city": self._city})]}
        # 城市不在表内（如"绿野仙踪"）：去掉疑问后缀后取中文名，
        # 让真实 geocode 工具返回 NOT_FOUND，演示诚实拒答
        clean = re.sub(r"(在哪里|在哪|在哪儿|呢|吗|怎么样|如何|是什么|是多少|多少)", "", question)
        m2 = re.search(r"[\u4e00-\u9fa5]{2,8}", clean)
        name = m2.group(0) if m2 else question.strip()[:6]
        return {"content": None, "tool_calls": [self._tool_call("geocode", {"city": name})]}

    def _react(self, last_tool_msg):
        """根据上一轮工具结果决定下一步：继续调工具 or 给最终答案。"""
        result = last_tool_msg["content"]

        if result.startswith("[ERROR]"):
            # —— 失败自愈 / 诚实拒答 ——
            if "[PARAM]" in result and self._typo_pending and self._city:
                # 模拟模型第一轮"手滑"传错坐标：本轮用 geocode 拿到的真实坐标修正重试
                lat, lon = self._coords or (self.city_db[self._city]["latitude"],
                                            self.city_db[self._city]["longitude"])
                self._typo_pending = False
                if self.verbose:
                    print(f"      ↻ [mock 自愈] 上轮坐标非法，已修正为 ({lat}, {lon}) 重试")
                return {"content": None, "tool_calls": [
                    self._tool_call("get_current_weather",
                                    {"latitude": lat, "longitude": lon})]}
            if "[NOT_FOUND]" in result and not self._geo_done:
                # geocode 找不到 → 不再瞎试，如实告知
                return {"content": f"抱歉，我查不到这个城市的位置信息。{result.split('] ', 1)[-1]}",
                        "tool_calls": None}
            return {"content": f"查询失败：{result}", "tool_calls": None}

        if not self._geo_done:
            # geocode 成功（JSON）→ 解析结构化输出，进入下一步链式调用
            loc = json.loads(result)
            self._coords = (loc["latitude"], loc["longitude"])
            self._geo_done = True
            if self._intent == "coords":
                return {"content": (f"根据 geocode 结果：{loc['name']}（{loc['country']} {loc['admin1']}）"
                                    f"的经纬度为 纬度 {loc['latitude']}、经度 {loc['longitude']}。"),
                        "tool_calls": None}
            return self._next_weather_call()

        if self._intent == "comfort" and "当前天气" in result:
            # 下游衍生链式：从 get_current_weather 的返回文本中提取温度/湿度/风速，
            # 作为 get_comfort_index 的参数（模拟真实 LLM 的"数据搬运"行为）
            m_t = re.search(r"温度：(-?\d+\.?\d*)°C", result)
            m_h = re.search(r"相对湿度：(-?\d+\.?\d*)%", result)
            m_w = re.search(r"风速：(-?\d+\.?\d*) km/h", result)
            if m_t and m_h and m_w:
                if self.verbose:
                    print(f"      ↻ [mock 提取] 从天气文本提取 温度={m_t.group(1)} 湿度={m_h.group(1)} 风速={m_w.group(1)}")
                return {"content": None, "tool_calls": [self._tool_call(
                    "get_comfort_index", {
                        "temperature": float(m_t.group(1)),
                        "humidity": float(m_h.group(1)),
                        "wind_speed": float(m_w.group(1)),
                    })]}

        # 天气/空气质量/舒适度数据已到手 → 组装最终回答
        return {"content": self._compose_answer(result), "tool_calls": None}

    def _next_weather_call(self):
        """按意图发起天气/空气质量工具调用；天气+空气同时问 → 一轮并行两个。"""
        lat, lon = self._coords
        if self._typo_pending:
            # 自愈演示：第一轮故意"手滑"传错坐标，工具返回 [ERROR][PARAM] 后由 _react 修正重试
            lat, lon = 999.0, 0.0
        calls = []
        if self._intent == "air":
            calls.append(self._tool_call("get_air_quality", {"latitude": lat, "longitude": lon}))
        elif self._intent == "both":
            calls.append(self._tool_call("get_current_weather", {"latitude": lat, "longitude": lon}))
            calls.append(self._tool_call("get_air_quality", {"latitude": lat, "longitude": lon}))
        elif self._intent == "forecast":
            calls.append(self._tool_call("get_daily_forecast", {"latitude": lat, "longitude": lon, "days": 3}))
        elif self._intent == "comfort":
            # 先拿当前天气，下一轮再从返回文本中提取参数调 get_comfort_index
            calls.append(self._tool_call("get_current_weather", {"latitude": lat, "longitude": lon}))
        elif self._intent == "coords":
            return {"content": f"该坐标的纬度为 {lat}、经度为 {lon}。", "tool_calls": None}
        else:  # current
            calls.append(self._tool_call("get_current_weather", {"latitude": lat, "longitude": lon}))
        return {"content": None, "tool_calls": calls}

    @staticmethod
    def _compose_answer(result: str) -> str:
        return f"根据工具查询结果：\n{result}"


# ── Agent Loop 主循环 ─────────────────────────────────────────────────────

class AgentLoop:
    """
    循环状态机：LLM 决策 → 宿主执行 → 结果回填 → 再决策……直到模型给最终答案。

    终止条件（三层）：
      1. 模型主动停：某轮不再输出 tool_calls → 该轮 content 即最终答案；
      2. 最大轮数：超过 MAX_STEPS 强制终止（防模型无限循环烧 token）；
      3. 死循环检测：连续多轮调用"同一工具 + 同一参数"视为死循环，强制终止。
    """

    def __init__(self, driver, model: str = "", max_steps: int = 8, verbose: bool = True):
        """
        driver: 两种之一
          - 真实模型: client.chat.completions（配合 model 参数）
          - MockPlanner 实例（.decide(messages)）
        """
        self.driver = driver
        self._model = model
        self.max_steps = max_steps
        self.verbose = verbose

    # ---- 决策层（区分真实 LLM 与 mock，输出统一规范化结构）----
    def _decide(self, messages):
        if isinstance(self.driver, MockPlanner):
            return self.driver.decide(messages)
        resp = self.driver.create(
            model=self._model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            } for tc in msg.tool_calls]
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return {"content": msg.content or "", "tool_calls": tool_calls, "usage": usage}

    def run(self, question: str):
        """
        执行一次完整的 Agent Loop。
        返回 {answer, rounds, tool_calls, usage, transcript, terminated_by, elapsed}。
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        t0 = time.time()
        tool_log, transcript = [], []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_call_key, dead_rounds = None, 0
        terminated_by = "model_stop"

        for step in range(1, self.max_steps + 1):
            decision = self._decide(messages)
            if decision.get("usage"):
                for k in usage_total:
                    usage_total[k] += decision["usage"].get(k, 0)

            calls = decision.get("tool_calls")
            if not calls:
                answer = decision.get("content") or ""
                if self.verbose:
                    print(f"  → [llm round {step}] 不再调用工具，输出最终回答")
                transcript.append({"round": step, "decision": "final_answer", "answer": answer})
                break

            # 本轮要调用工具：先把 assistant 消息回填（tool_call_id 必须对应）
            messages.append({
                "role": "assistant",
                "content": decision.get("content") or "",
                "tool_calls": [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                } for c in calls],
            })

            # 逐个执行（一轮多个 = 并行调用），结果以 role=tool 回填
            round_log = {"round": step, "tool_calls": []}
            for c in calls:
                name, args_str = c["name"], c["arguments"]
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_DISPATCH.get(name)
                if fn is None:
                    result = f"[ERROR][UNKNOWN_TOOL] 未知工具：{name}"
                else:
                    try:
                        result = fn(**args)
                    except TypeError as e:
                        result = f"[ERROR][PARAM] 工具 {name} 参数错误：{e}"
                    except Exception as e:
                        result = f"[ERROR][EXEC] 工具 {name} 执行失败：{e}"
                preview = (result or "")[:110].replace("\n", " ")
                if self.verbose:
                    print(f"  → [tool round {step}] {name}({args})")
                    print(f"    ↩ {preview}{'…' if len(result or '') > 110 else ''}")
                messages.append({
                    "role": "tool", "tool_call_id": c["id"], "content": result,
                })
                tool_log.append({"name": name, "args": args})
                round_log["tool_calls"].append({"name": name, "args": args, "result": result})
            transcript.append(round_log)

            # 死循环检测：连续两轮调用集合完全一致 → 强制终止
            call_key = frozenset((c["name"], c["arguments"]) for c in calls)
            dead_rounds = dead_rounds + 1 if call_key == last_call_key else 0
            last_call_key = call_key
            if dead_rounds >= 2:
                answer = (f"（检测到连续 {dead_rounds + 1} 轮重复调用相同工具参数，"
                          f"判定为死循环，已强制终止）")
                terminated_by = "dead_loop"
                if self.verbose:
                    print(f"  → [guard] 死循环检测触发，终止循环")
                transcript.append({"round": step, "decision": "dead_loop"})
                break
        else:
            # for 正常耗尽 = 达到最大轮数
            answer = f"（达到最大循环轮数 {self.max_steps}，模型仍未给出最终回答，已强制终止）"
            terminated_by = "max_steps"
            transcript.append({"round": self.max_steps, "decision": "max_steps"})

        return {
            "answer": answer,
            "rounds": len(transcript),
            "tool_calls": tool_log,
            "usage": usage_total,
            "transcript": transcript,
            "terminated_by": terminated_by,
            "elapsed": time.time() - t0,
        }


# ── 入口 ─────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德今天的天气怎么样？",                # 链式：geocode → get_current_weather
    "北京未来3天的天气怎么样？",              # 链式：geocode → get_daily_forecast
    "福州的空气质量怎么样？",                # 链式：geocode → get_air_quality
    "宁德今天体感怎么样？",                  # 下游衍生链式：geocode → 天气 → 提取参数 → 舒适度
    "经度119.52、纬度26.66 的当前天气和空气质量？",  # 并行：一轮两个工具
    "北惊的天气怎么样？",                   # 失败自愈：geocode 拼写错误 → 模型修正重试
    "绿野仙踪在哪里？",                     # 诚实拒答：城市不存在 → 不编造
]

MOCK_DEMO_QUESTIONS = [
    "宁德今天的天气怎么样？",
    "北京未来3天的天气怎么样？",
    "宁德今天体感舒适吗？",                   # 下游衍生链式（模型提取参数）
    "经度119.52、纬度26.66 的当前天气和空气质量？",     # 并行调用
    "北京的天气怎么样（模拟第一轮手滑传错坐标）",        # 失败自愈：PARAM → 修正重试
    "绿野仙踪在哪里？",                           # 诚实拒答：NOT_FOUND
]


def main():
    parser = argparse.ArgumentParser(
        description="作业：天气工具调用改造为循环调用（Agent Loop）")
    parser.add_argument("-q", "--question", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟决策器（无需 API Key，离线演示循环机制）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-steps", type=int, default=8, help="循环最大轮数（默认 8）")
    parser.add_argument("--transcript", metavar="FILE",
                        help="把每轮循环轨迹（JSONL）导出到指定文件")
    parser.add_argument("--quiet", action="store_true", help="少输出")
    args = parser.parse_args()

    # 组装驱动：mock 或真实模型
    if args.mock:
        driver = MockPlanner(TOOLS_SCHEMA, verbose=not args.quiet)
        model = "(mock)"
        print(f"[Weather Agent Loop] driver=mock 无需 API Key\n")
    else:
        client, model = build_client(args.provider)
        driver = client.chat.completions
        print(f"[Weather Agent Loop] provider={args.provider} model={model}\n")

    agent = AgentLoop(driver, model=model, max_steps=args.max_steps, verbose=not args.quiet)

    questions = (MOCK_DEMO_QUESTIONS if args.mock else DEMO_QUESTIONS) if args.demo \
        else ([args.question] if args.question else (MOCK_DEMO_QUESTIONS if args.mock else DEMO_QUESTIONS[:1]))

    fp = open(args.transcript, "w", encoding="utf-8") if args.transcript else None
    try:
        for i, q in enumerate(questions, 1):
            print("=" * 64)
            print(f"Q{i}：{q}")
            print("=" * 64)
            result = agent.run(q)
            print("\n最终回答：")
            print(result["answer"])
            u = result["usage"]
            print(f"\n（{result['terminated_by']}：工具调用 {len(result['tool_calls'])} 次，"
                  f"循环 {result['rounds']} 轮，token {u['total_tokens']}，"
                  f"耗时 {result['elapsed']:.1f}s）\n")
            if fp:
                fp.write(json.dumps({"question": q, **result}, ensure_ascii=False) + "\n")
                fp.flush()
    finally:
        if fp:
            fp.close()
            print(f"轨迹已导出到 {args.transcript}")


if __name__ == "__main__":
    main()
