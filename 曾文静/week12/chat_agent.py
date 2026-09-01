"""
chat_agent.py — 多轮对话 Agent（ChatAgent）

作业核心：给 week11 的「单轮 Agent Loop」增加多轮对话能力。

week11（agent_loop.py）的问题：
   每次 run(question) 都从 [system, user] 开始 —— 第二轮提问时，
   模型完全不记得第一轮查过什么，追问「那空气质量呢」只能靠猜。

本作业的改造（与常见作业的差异点）：
   1. 记忆层接管上下文：每轮开始从 memory.py 的 MemoryManager 读三层记忆
      （滚动摘要 + 关键事实 + 最近窗口问答）组装 messages；
      每轮结束把「问题 + 最终回答 + 新事实」写回记忆。见 memory.py。
   2. 循环本身不变：单轮 ReAct 循环（Thought→Action→Observation）原样保留，
      多轮 = 「读记忆 → 跑循环 → 写记忆」的外层包装 —— 记忆与循环解耦。
   3. 双驱动：真实 LLM（OpenAI 兼容接口）与 StatefulMockPlanner（带跨轮记忆
      的规则模拟器）走同一套 Agent 逻辑。mock 不烧 token 也能完整验证
      「追问复用 / 事实回答 / 摘要压缩 / 会话恢复」四条多轮能力。
   4. 三层终止保护沿用 week11：模型主动停 / 最大步数 / 死循环检测。

StatefulMockPlanner 与 week11 MockPlanner 的区别（这是多轮的关键）：
   week11 的 mock 每题重置决策状态，是「无记忆的模拟器」；
   本文件的 mock 会解析系统提示里的 [已掌握的关键事实] 与 [此前对话摘要]，
   能处理「那空气质量呢」「明天呢」「北京和上海哪个更冷」这类跨轮追问 ——
   相当于用规则把「记忆如何被消费」也模拟了一遍，离线即可验证记忆层有效。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional

from openai import OpenAI

# 让同目录的 weather_tools / memory 可被 import（直接 python 运行也能找到）
sys.path.insert(0, str(Path(__file__).parent))

from weather_tools import TOOLS  # noqa: E402
from memory import MemoryManager, Turn  # noqa: E402

# ── LLM 配置（与 week11 同一套 PROVIDERS 结构）───────────────────────────────

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


# ── 工具 Schema（与 week11 agent_loop.py 完全一致，保证工具口径不变）──────────

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
]

TOOL_DISPATCH = dict(TOOLS)  # 工具名 → 后端函数（业务逻辑在 weather_tools.py）


# ── 离线工具后端（MockToolBackend）──────────────────────────────────────────
# 真实工具依赖 Open-Meteo 网络接口，演示/验证时网络抖动会导致结果不可复现。
# 与 StatefulMockPlanner 配套，本后端用内置固定数据集仿真 4 个工具的返回
# （含 [ERROR] 协议），让「多轮记忆验证」完全离线、确定性可复现：
#   同一问题在任何时间运行，工具返回与判定结果完全一致。
# 返回格式与 weather_tools.py 完全一致（geocode 的 JSON 结构、文本行格式），
# 因此记忆层的事实抽取、mock 决策逻辑都不需要感知真假工具的区别。

MOCK_CITY_DATA = {
    "宁德": {"name": "宁德", "country": "中国", "admin1": "福建省",
             "latitude": 26.66, "longitude": 119.52,
             "weather": "小雨", "temp": 18.0, "humidity": 85, "wind": 5.2,
             "aqi": 52, "level": "良", "pm25": 18, "pm10": 30, "o3": 60, "no2": 15,
             "fc": [("小雨", 20.0, 15.0, 2.0), ("中雨", 19.0, 14.0, 8.0), ("阴天", 22.0, 16.0, 0.0)]},
    "北京": {"name": "北京", "country": "中国", "admin1": "北京市",
             "latitude": 39.90, "longitude": 116.40,
             "weather": "晴", "temp": 12.0, "humidity": 30, "wind": 8.0,
             "aqi": 88, "level": "良", "pm25": 40, "pm10": 70, "o3": 110, "no2": 35,
             "fc": [("晴", 15.0, 5.0, 0.0), ("多云", 14.0, 6.0, 0.0), ("小雨", 12.0, 4.0, 3.0)]},
    "上海": {"name": "上海", "country": "中国", "admin1": "上海市",
             "latitude": 31.23, "longitude": 121.47,
             "weather": "多云", "temp": 22.0, "humidity": 65, "wind": 6.0,
             "aqi": 60, "level": "良", "pm25": 25, "pm10": 45, "o3": 80, "no2": 25,
             "fc": [("多云", 24.0, 19.0, 0.0), ("雷阵雨", 23.0, 18.0, 6.0), ("小雨", 22.0, 17.0, 4.0)]},
    "深圳": {"name": "深圳", "country": "中国", "admin1": "广东省",
             "latitude": 22.54, "longitude": 114.06,
             "weather": "大雨", "temp": 27.0, "humidity": 88, "wind": 10.0,
             "aqi": 42, "level": "优", "pm25": 12, "pm10": 22, "o3": 45, "no2": 12,
             "fc": [("大雨", 28.0, 24.0, 15.0), ("中雨", 27.0, 23.0, 9.0), ("小雨", 29.0, 24.0, 3.0)]},
    "广州": {"name": "广州", "country": "中国", "admin1": "广东省",
             "latitude": 23.13, "longitude": 113.26,
             "weather": "雷阵雨", "temp": 29.0, "humidity": 82, "wind": 7.0,
             "aqi": 55, "level": "良", "pm25": 20, "pm10": 35, "o3": 70, "no2": 18,
             "fc": [("雷阵雨", 30.0, 25.0, 12.0), ("小雨", 29.0, 24.0, 3.0), ("多云", 31.0, 25.0, 0.0)]},
}

import datetime as _dt  # noqa: E402


def _mock_date(offset: int) -> str:
    return (_dt.date.today() + _dt.timedelta(days=offset)).isoformat()


def _mock_find_city(lat: float, lon: float) -> str | None:
    for name, d in MOCK_CITY_DATA.items():
        if abs(d["latitude"] - float(lat)) < 0.05 and abs(d["longitude"] - float(lon)) < 0.05:
            return name
    return None


def mock_geocode(city: str) -> str:
    city = (city or "").strip()
    d = MOCK_CITY_DATA.get(city)
    if d is None:
        return (f"[ERROR][NOT_FOUND] 未找到城市 '{city}'（离线数据集仅支持："
                f"{'、'.join(MOCK_CITY_DATA)}）")
    return json.dumps({"name": d["name"], "country": d["country"],
                       "admin1": d["admin1"], "latitude": d["latitude"],
                       "longitude": d["longitude"]}, ensure_ascii=False)


def mock_current_weather(latitude: float, longitude: float) -> str:
    err = _check_coords_mock(latitude, longitude)
    if err:
        return err
    name = _mock_find_city(latitude, longitude)
    if name is None:
        return "[ERROR][PARAM] 坐标不在离线数据集内，请使用 geocode 查询已知城市"
    d = MOCK_CITY_DATA[name]
    return (f"当前天气：{d['weather']}\n温度：{d['temp']}°C\n"
            f"相对湿度：{d['humidity']}%\n风速：{d['wind']} km/h")


def mock_daily_forecast(latitude: float, longitude: float, days: int = 3) -> str:
    err = _check_coords_mock(latitude, longitude)
    if err:
        return err
    try:
        days = max(1, min(7, int(days)))
    except (TypeError, ValueError):
        return f"[ERROR][PARAM] days 必须是整数（1~7），收到 {days!r}。"
    name = _mock_find_city(latitude, longitude)
    if name is None:
        return "[ERROR][PARAM] 坐标不在离线数据集内，请使用 geocode 查询已知城市"
    d = MOCK_CITY_DATA[name]
    lines = [f"未来{days}天预报："]
    for i in range(days):
        desc, hi, lo, rain = d["fc"][i % len(d["fc"])]
        lines.append(f"  {_mock_date(i)}：{desc}，最高 {hi}°C / 最低 {lo}°C，降水 {rain} mm")
    return "\n".join(lines)


def mock_air_quality(latitude: float, longitude: float) -> str:
    err = _check_coords_mock(latitude, longitude)
    if err:
        return err
    name = _mock_find_city(latitude, longitude)
    if name is None:
        return "[ERROR][PARAM] 坐标不在离线数据集内，请使用 geocode 查询已知城市"
    d = MOCK_CITY_DATA[name]
    return (f"空气质量指数 AQI：{d['aqi']}（{d['level']}）\n"
            f"PM2.5：{d['pm25']} µg/m³\nPM10：{d['pm10']} µg/m³\n"
            f"臭氧 O₃：{d['o3']} µg/m³\n二氧化氮 NO₂：{d['no2']} µg/m³")


def _check_coords_mock(latitude: float, longitude: float) -> str | None:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return f"[ERROR][PARAM] 坐标必须是数字，收到 latitude={latitude!r}, longitude={longitude!r}"
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return (f"[ERROR][PARAM] 坐标越界：纬度 {lat}、经度 {lon}。"
                f"纬度范围 [-90, 90]，经度范围 [-180, 180]，注意经纬度不要写反。")
    return None


MOCK_TOOL_DISPATCH = {
    "geocode": mock_geocode,
    "get_current_weather": mock_current_weather,
    "get_daily_forecast": mock_daily_forecast,
    "get_air_quality": mock_air_quality,
}

SYSTEM_PROMPT = (
    "你是一名智能天气助手，拥有 4 个专业工具：geocode（城市名→经纬度）、"
    "get_current_weather（当前天气）、get_daily_forecast（未来多天预报）、"
    "get_air_quality（空气质量）。\n\n"
    "工作流程：\n"
    "- 用户只问城市经纬度 → 只调 geocode；\n"
    "- 用户问城市天气/预报/空气质量 → 先 geocode 拿经纬度，再按需调用对应工具；\n"
    "- 用户直接给出经纬度 → 跳过 geocode，直接调用天气/空气质量工具；\n"
    "- 需要同时知道天气和空气质量时，可以在同一轮并行调用两个工具。\n\n"
    "多轮对话规则（本周作业重点）：\n"
    "- 系统提示末尾的 [此前对话摘要] 与 [已掌握的关键事实] 是历史记忆，"
    "必须优先利用：用户追问「那/它/这个城市」「空气质量呢」「明天呢」时，"
    "指的是最近查过的城市或指标，不要重复询问；\n"
    "- 记忆里已有的事实（如某城市坐标、温度、AQI）直接引用，不要重复调用工具查询；\n"
    "- 回答要自然衔接上一轮，例如先复述已知信息再补新信息。\n\n"
    "错误处理：\n"
    "- 工具返回 [ERROR][NOT_FOUND]/[ERROR][PARAM] 时，先根据错误提示修正参数重试一次；\n"
    "- 重试仍失败，如实告知用户原因，绝不编造数据；\n"
    "- 严格依据工具返回的真实数据回答，不要猜测。"
)


# ── 记忆回调 1：规则式事实抽取（真实/mock 共用，不依赖 LLM）───────────────────
# 从本轮 tool_log 里解析结构化事实：geocode 的 JSON 拿坐标，天气/AQI/预报文本拿数值。
# 事实格式统一为「城市 + 指标 + 值」，便于 mock 与真实模型共同消费。

_CITY_RE = re.compile(r'"name":\s*"([^"]+)"')
_COORD_RE = re.compile(r'"latitude":\s*([\d.+-]+).*?"longitude":\s*([\d.+-]+)')
_TEMP_RE = re.compile(r"温度：([\d.+-]+)°C")
_WEATHER_RE = re.compile(r"当前天气：(\S+)")
_AQI_RE = re.compile(r"空气质量指数 AQI：(\d+)（([^）]+)）")
_FORECAST_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})：(\S+)，最高 ([\d.+-]+)°C / 最低 ([\d.+-]+)°C")


def rule_fact_extractor(turn: Turn, tool_log: list, known_facts: Optional[list] = None) -> List[str]:
    """
    从本轮 tool_log 里解析结构化事实。城市归属有两个来源：
      1. 本轮 geocode 的 JSON（name 字段）；
      2. 记忆里已有的「{城市} 坐标 (lat, lon)」事实（坐标反查城市）——
         这正好覆盖「坐标复用、跳过 geocode」的多轮场景，否则复用坐标的
         轮次（如"那空气质量呢"）抽不出可归属的事实。
    """
    facts: List[str] = []
    coord_to_city = {}
    for f in (known_facts or []):
        m = re.match(r"^(.+?) 坐标 \(([\d.+-]+), ([\d.+-]+)\)", f)
        if m:
            coord_to_city[(m.group(2), m.group(3))] = m.group(1)
    last_city: Optional[str] = None
    for item in tool_log:
        result = str(item.get("result", ""))
        if item.get("name") == "geocode":
            m_city, m_coord = _CITY_RE.search(result), _COORD_RE.search(result)
            if m_city and m_coord:
                last_city = m_city.group(1)
                coord_to_city[(m_coord.group(1), m_coord.group(2))] = last_city
                facts.append(f"{last_city} 坐标 ({m_coord.group(1)}, {m_coord.group(2)})")
            continue
        args = item.get("args") or {}
        if last_city is None and "latitude" in args and "longitude" in args:
            key = (str(args["latitude"]), str(args["longitude"]))
            last_city = coord_to_city.get(key)
        if last_city is None:
            continue  # 无法归属城市的事实不记录（宁缺毋滥，防止事实表混入脏数据）
        if item.get("name") == "get_current_weather":
            m_t, m_w = _TEMP_RE.search(result), _WEATHER_RE.search(result)
            if m_w:
                facts.append(f"{last_city} 当前天气 {m_w.group(1)}")
            if m_t:
                facts.append(f"{last_city} 当前温度 {m_t.group(1)}°C")
        elif item.get("name") == "get_air_quality":
            m_a = _AQI_RE.search(result)
            if m_a:
                facts.append(f"{last_city} AQI {m_a.group(1)}（{m_a.group(2)}）")
        elif item.get("name") == "get_daily_forecast":
            for line in result.splitlines():
                m_f = _FORECAST_RE.match(line)
                if m_f:
                    facts.append(
                        f"{last_city} {m_f.group(1)} 预报 {m_f.group(2)} "
                        f"最高 {m_f.group(3)}°C / 最低 {m_f.group(4)}°C")
    return facts


# ── 记忆回调 2：滚动摘要压缩器 ────────────────────────────────────────────────
# 真实模式：LLM 压缩；mock 模式：规则模板（保留轮次编号与问答骨架）。

def make_llm_summarizer(client, model: str):
    """返回 summarizer(summary, evicted_turns) -> str（LLM 滚动压缩）。"""
    def _summarize(old_summary: str, evicted: List[Turn]) -> str:
        turns_text = "\n".join(
            f"第{t.turn}轮 用户问：{t.question}\n第{t.turn}轮 助手答：{t.answer[:200]}"
            for t in evicted
        )
        prompt = (
            "你是一个对话记忆压缩器。把「旧摘要」与「新发生的对话轮次」合并压缩成"
            "一段 150 字以内的要点，保留：用户查过的城市/指标、关键数值、结论。"
            "不要编造内容。\n\n"
            f"[旧摘要]\n{old_summary or '（无）'}\n\n"
            f"[新对话轮次]\n{turns_text or '（无）'}\n\n"
            "压缩结果："
        )
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        return (resp.choices[0].message.content or "").strip()[:500]
    return _summarize


def make_rule_summarizer(max_entries: int = 6, max_chars: int = 400):
    """mock 模式：把被挤出的轮次按「问→答」骨架追加到摘要。
    以「；」分隔条目，超限时从最旧条目开始丢弃（保证条目标题完整、
    编号正确，不会像直接截断字符串那样切出半截）。"""
    def _summarize(old_summary: str, evicted: List[Turn]) -> str:
        entries = [e for e in old_summary.split("；") if e] if old_summary else []
        for t in evicted:
            q = t.question.replace("\n", " ")[:24]
            a = t.answer.replace("\n", " ")[:64]
            entries.append(f"第{t.turn}轮 问「{q}」→答「{a}」")
        # 从最旧丢弃：条数超限或总长超限
        while len(entries) > max_entries or sum(len(e) for e in entries) > max_chars:
            entries.pop(0)
        return "；".join(entries)
    return _summarize


# ── Mock 驱动：带跨轮记忆的规则模拟器 ────────────────────────────────────────

class StatefulMockPlanner:
    """
    用规则模拟 LLM 的决策，但【跨轮有记忆】：它和真实模型看到的是同一份
    上下文（系统提示里的摘要块 + 事实块 + 窗口问答），因此能验证：
      - 追问复用：第二轮问「那空气质量呢」→ 从事实里找到上一轮城市坐标，
        直接调 air quality，不再重复 geocode（工具调用数下降 = 记忆生效）；
      - 纯记忆回答：两城温度都在事实里时，问「哪个更冷」→ 不调任何工具，
        直接基于事实回答；
      - 上下文组装正确性：事实块/摘要块确实被下游「模型」读取到了。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.city_db = {
            "宁德": {"name": "宁德", "country": "中国", "admin1": "福建省",
                     "latitude": 26.66, "longitude": 119.52},
            "北京": {"name": "北京", "country": "中国", "admin1": "北京市",
                     "latitude": 39.90, "longitude": 116.40},
            "上海": {"name": "上海", "country": "中国", "admin1": "上海市",
                     "latitude": 31.23, "longitude": 121.47},
            "深圳": {"name": "深圳", "country": "中国", "admin1": "广东省",
                     "latitude": 22.54, "longitude": 114.06},
            "广州": {"name": "广州", "country": "中国", "admin1": "广东省",
                     "latitude": 23.13, "longitude": 113.26},
        }
        # 本轮决策状态（每轮重置，与 week11 一致）
        self._intent = None
        self._city = None
        self._coords = None
        self._geo_done = False
        self._call_seq = 0

    # ---- 与真实 LLM 同构的入口 ----
    def decide(self, messages):
        last = messages[-1]
        if last["role"] == "user":
            return self._plan(messages)
        return self._react(last)

    def _tool_call(self, name, args):
        self._call_seq += 1
        return {"id": f"call_mock_{self._call_seq}", "name": name,
                "arguments": json.dumps(args, ensure_ascii=False)}

    # ---- 从上下文里读记忆 ----
    def _read_context(self, messages) -> dict:
        """解析系统提示里的记忆块 + 最近问答，返回 {facts: [...], recent_questions: [...]}。"""
        facts, recent_questions = [], []
        for m in messages:
            if m["role"] == "system":
                for line in m.get("content", "").splitlines():
                    if line.startswith("[已掌握的关键事实]"):
                        facts = [f.strip() for f in
                                 line.replace("[已掌握的关键事实]", "", 1).split("；") if f.strip()]
            elif m["role"] == "user" and "（第" in m.get("content", ""):
                recent_questions.append(m["content"])
        return {"facts": facts, "recent_questions": recent_questions}

    def _city_in_question(self, q: str) -> Optional[str]:
        for name in self.city_db:
            if name in q:
                return name
        return None

    def _last_city_from_facts(self, facts: List[str]) -> Optional[str]:
        """从事实表里找最近提到的城市（事实是追加序，取最后一个带坐标的城市）。"""
        for f in reversed(facts):
            m = re.match(r"^(.+?) 坐标 \(", f)   # 事实格式："{城市} 坐标 (lat, lon)"
            if m:
                return m.group(1)
        return None

    def _facts_for(self, facts: List[str], city: str) -> List[str]:
        return [f for f in facts if f.startswith(city)]

    def _plan(self, messages):
        # 每题独立：重置本轮决策状态（记忆在上下文里，天然跨轮）
        self._intent = None
        self._city = None
        self._coords = None
        self._geo_done = False
        self._call_seq = 0

        question = messages[-1]["content"]
        self._last_question = question   # 供 _answer_compare 判断「更热/更冷」语义
        ctx = self._read_context(messages)
        facts = ctx["facts"]

        # —— 指代消解：问题里没有城市名，但从记忆里能找到「最近查过的城市」 ——
        city = self._city_in_question(question)
        has_intent = any(k in question for k in
                         ("天气", "温度", "空气", "预报", "未来", "几天", "AQI", "坐标", "经纬度"))
        # 纯追问（"明天呢？"）也算意图：结尾是"呢/？"且没有新城市名
        looks_followup = (question.rstrip("？? ") .endswith("呢")) and len(question) <= 12
        if city is None and (has_intent or looks_followup):
            city = self._last_city_from_facts(facts)
            if city and self.verbose:
                print(f"      ↻ [mock 记忆] 问题未含城市名，指代消解为最近查过的「{city}」")

        # —— 纯记忆回答：对比类问题且两城数据都在事实里 → 不调工具 ——
        compare = self._parse_compare(question)
        if compare:
            return self._answer_compare(compare, facts)

        if city is None:
            return {"content": "请问你要查哪个城市的天气/空气质量？告诉我城市名即可。",
                    "tool_calls": None}

        # —— 意图识别（沿用 week11；纯追问默认按"当前天气"处理，
        #     但提到 明天/未来 时按预报处理） ——
        if "空气" in question or "AQI" in question:
            self._intent = "air"
        elif any(k in question for k in ("预报", "未来", "明天", "后天", "几天")):
            self._intent = "forecast"
        elif "坐标" in question or "经纬度" in question:
            self._intent = "coords"
        else:
            self._intent = "current"
        self._city = city

        # —— 事实复用：该城市坐标已在记忆里 → 跳过 geocode，直接查天气 ——
        for f in facts:
            if f.startswith(f"{city} 坐标"):
                m = re.search(r"\(([\d.+-]+), ([\d.+-]+)\)", f)
                if m:
                    self._coords = (float(m.group(1)), float(m.group(2)))
                    self._geo_done = True
                    if self.verbose:
                        print(f"      ↻ [mock 记忆] 复用「{city} 坐标」事实，跳过 geocode")
                    return self._next_weather_call()

        return {"content": None,
                "tool_calls": [self._tool_call("geocode", {"city": city})]}

    def _parse_compare(self, q: str) -> Optional[tuple]:
        """识别「A和B哪个更冷/更热/更适合…」→ 返回 (city_a, city_b)。"""
        if not any(k in q for k in ("哪个", "谁更", "对比", "比较")):
            return None
        cities = [c for c in self.city_db if c in q]
        if len(cities) >= 2:
            return (cities[0], cities[1])
        return None

    def _answer_compare(self, pair: tuple, facts: List[str]):
        a, b = pair
        fa, fb = self._facts_for(facts, a), self._facts_for(facts, b)
        # 空气对比：问句中含 空气/AQI/污染 → 比 AQI（越低越好）
        is_air = any(k in self._last_question for k in ("空气", "AQI", "污染"))
        key = "AQI" if is_air else "当前温度"
        ta = next((f for f in fa if key in f), None)
        tb = next((f for f in fb if key in f), None)
        if ta and tb:
            if is_air:
                va = int(re.search(r"AQI (\d+)", ta).group(1))
                vb = int(re.search(r"AQI (\d+)", tb).group(1))
                pick = a if va < vb else b
                verb = "空气质量更好"
                if self.verbose:
                    print(f"      ↻ [mock 记忆] AQI 对比：{a} {va} vs {b} {vb}（0 次工具调用）")
                return {"content": f"根据此前查询的事实：{a} AQI {va}，{b} AQI {vb}，"
                                   f"{pick} 的{verb}。", "tool_calls": None}
            va = float(re.search(r"([\d.+-]+)°C", ta).group(1))
            vb = float(re.search(r"([\d.+-]+)°C", tb).group(1))
            want_hotter = any(k in self._last_question for k in ("更热", "更暖", "热一些"))
            pick = (a if va > vb else b) if want_hotter else (a if va < vb else b)
            verb = "更热" if want_hotter else "更冷"
            if self.verbose:
                print(f"      ↻ [mock 记忆] 温度对比：{a} {va}°C vs {b} {vb}°C（0 次工具调用）")
            return {"content": f"根据此前查询的事实：{a} 当前 {va}°C，{b} 当前 {vb}°C，"
                               f"今天 {pick} {verb}。", "tool_calls": None}
        return {"content": f"我还不知道 {a} 和 {b} 的完整数据，先分别查一下再对比。",
                "tool_calls": None}

    def _react(self, last_tool_msg):
        result = last_tool_msg["content"]
        if result.startswith("[ERROR]"):
            if "[PARAM]" in result and self._coords and self._geo_done:
                # 自愈：坐标非法时用事实/数据库里的正确坐标重试
                if self.verbose:
                    print("      ↻ [mock 自愈] 上轮坐标非法，用已知正确坐标重试")
                return self._next_weather_call()
            if "[NOT_FOUND]" in result:
                return {"content": f"抱歉，我查不到这个城市的位置信息。{result.split('] ', 1)[-1]}",
                        "tool_calls": None}
            return {"content": f"查询失败：{result}", "tool_calls": None}
        if not self._geo_done:
            loc = json.loads(result)
            self._coords = (loc["latitude"], loc["longitude"])
            self._geo_done = True
            if self._intent == "coords":
                return {"content": (f"根据 geocode 结果：{loc['name']}（{loc['country']} {loc['admin1']}）"
                                    f"的经纬度为 纬度 {loc['latitude']}、经度 {loc['longitude']}。"),
                        "tool_calls": None}
            return self._next_weather_call()
        return {"content": f"查询成功，工具返回的数据如下：\n{result}", "tool_calls": None}

    def _next_weather_call(self):
        lat, lon = self._coords
        if self._intent == "air":
            calls = [self._tool_call("get_air_quality", {"latitude": lat, "longitude": lon})]
        elif self._intent == "forecast":
            calls = [self._tool_call("get_daily_forecast",
                                     {"latitude": lat, "longitude": lon, "days": 3})]
        elif self._intent == "coords":
            return {"content": f"该坐标的纬度为 {lat}、经度为 {lon}。", "tool_calls": None}
        else:
            calls = [self._tool_call("get_current_weather", {"latitude": lat, "longitude": lon})]
        return {"content": None, "tool_calls": calls}


# ── 多轮对话 Agent ─────────────────────────────────────────────────────────

class ChatAgent:
    """
    多轮对话 Agent：外层管记忆，内层跑 ReAct 循环。

    一轮 chat(question) 的生命周期：
      读记忆   memory.build_context()  →  组装 [system+摘要+事实, *窗口问答, user]
      跑循环   [LLM决策 → 执行工具 → 回填结果] × N → 最终回答
              （终止保护三层：模型主动停 / 最大步数 / 死循环检测，沿用 week11）
      写记忆   memory.end_turn()  →  追加轮次、抽取新事实、必要时滚动压缩

    yield 结构（与 week11 对齐，便于 REPL/脚本统一消费）：
      {"type": "action", "step", "action", "action_input", "observation"}
      {"type": "final",  "step", "answer", "turn", "usage", "memory"}
      {"type": "max_steps" | "dead_loop", "step", "answer"}
    """

    def __init__(self, driver, model: str = "", memory: Optional[MemoryManager] = None,
                 max_steps: int = 8, verbose: bool = True,
                 tool_dispatch: Optional[dict] = None):
        self.driver = driver
        self._model = model
        self.memory = memory or MemoryManager()
        self.max_steps = max_steps
        self.verbose = verbose
        # 工具后端可插拔：真实（weather_tools）或离线仿真（MOCK_TOOL_DISPATCH）
        self.tool_dispatch = tool_dispatch or TOOL_DISPATCH

    # ---- 决策层（真实 LLM 与 mock 统一规范化输出） ----
    def _decide(self, messages):
        if isinstance(self.driver, StatefulMockPlanner):
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
                "id": tc.id, "name": tc.function.name,
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

    def chat(self, question: str) -> Generator[dict, None, None]:
        """处理一轮用户输入（读记忆 → ReAct 循环 → 写记忆），逐步 yield。"""
        # ① 读记忆：组装上下文
        messages = self.memory.build_context(question, SYSTEM_PROMPT)

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

            messages.append({
                "role": "assistant",
                "content": decision.get("content") or "",
                "tool_calls": [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                } for c in calls],
            })

            round_log = {"round": step, "tool_calls": []}
            for c in calls:
                name, args_str = c["name"], c["arguments"]
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                fn = self.tool_dispatch.get(name)
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
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                tool_log.append({"name": name, "args": args, "result": result})
                round_log["tool_calls"].append({"name": name, "args": args, "result": result})

                yield {
                    "type": "action", "step": step,
                    "action": name, "action_input": args,
                    "observation": result,
                }
            transcript.append(round_log)

            # 死循环检测（沿用 week11）：连续两轮调用集合完全一致 → 强制终止
            call_key = frozenset((c["name"], c["arguments"]) for c in calls)
            dead_rounds = dead_rounds + 1 if call_key == last_call_key else 0
            last_call_key = call_key
            if dead_rounds >= 2:
                answer = (f"（检测到连续 {dead_rounds + 1} 轮重复调用相同工具参数，"
                          f"判定为死循环，已强制终止）")
                terminated_by = "dead_loop"
                transcript.append({"round": step, "decision": "dead_loop"})
                break
        else:
            answer = f"（达到最大循环轮数 {self.max_steps}，模型仍未给出最终回答，已强制终止）"
            terminated_by = "max_steps"
            transcript.append({"round": self.max_steps, "decision": "max_steps"})

        # ② 写记忆：追加轮次 + 抽取事实 + 可能触发滚动压缩
        turn = self.memory.end_turn(question, answer, tool_log)

        yield {
            "type": "final",
            "step": len(transcript),
            "answer": answer,
            "terminated_by": terminated_by,
            "turn": turn.to_dict(),
            "usage": usage_total,
            "memory": self.memory.stats(),
            "elapsed": time.time() - t0,
        }


def build_client(provider: str = "deepseek"):
    """构造真实 LLM 驱动；未配置 API Key 时给出提示。"""
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY（或使用 --mock 模拟驱动）", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def build_agent(provider: str = "deepseek", model: str = "", mock: bool = False,
                max_steps: int = 8, verbose: bool = True,
                window_turns: int = 6, token_budget: int = 4000,
                mock_tools: bool = True) -> ChatAgent:
    """
    统一装配：驱动 + 记忆 + 记忆回调（摘要器按模式切换，事实抽取共用规则版）。

    mock=True 时默认同时启用离线工具后端（mock_tools=True），
    整个链路（LLM 决策 + 工具执行）都不需要网络，验证结果可复现。
    真实模型配 --mock-tools=False 时走 Open-Meteo 真实天气接口。
    """
    if mock:
        driver = StatefulMockPlanner(verbose=verbose)
        model_name = "(mock)"
        summarizer = make_rule_summarizer()
        tool_dispatch = MOCK_TOOL_DISPATCH if mock_tools else TOOL_DISPATCH
    else:
        client, model_name = build_client(provider)
        model_name = model or model_name
        driver = client.chat.completions
        summarizer = make_llm_summarizer(client, model_name)
        # 真实模型也可以配离线工具（--mock-tools）：验证 LLM 对记忆块的消费，
        # 同时不依赖网络天气接口，结果可复现
        tool_dispatch = MOCK_TOOL_DISPATCH if mock_tools else TOOL_DISPATCH

    memory = MemoryManager(
        window_turns=window_turns,
        token_budget=token_budget,
        summarizer=summarizer,
        fact_extractor=rule_fact_extractor,
    )
    return ChatAgent(driver, model=model_name, memory=memory,
                     max_steps=max_steps, verbose=verbose,
                     tool_dispatch=tool_dispatch)


if __name__ == "__main__":
    # 快速冒烟：python chat_agent.py --mock "宁德天气" "那空气质量呢"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("questions", nargs="+")
    args = ap.parse_args()
    agent = build_agent(provider=args.provider, model=args.model, mock=args.mock)
    for q in args.questions:
        print(f"\n===== 你：{q} =====")
        for step in agent.chat(q):
            if step["type"] == "final":
                print(f"✅ {step['answer']}")
