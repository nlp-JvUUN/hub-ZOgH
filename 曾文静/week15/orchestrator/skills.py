"""
skills.py — worker 技能注册表（对应 week13 SkillFlow 的 L1 技能清单）

核心设计（与"硬编码场景工具"的本质区别）：
  - 主 agent 不直接拥有任何业务工具，它只认识两个编排工具（list_skills / dispatch_workers）；
  - 所有业务能力以「技能」形式注册在 SKILL_REGISTRY 里，每个技能声明：
      名字 / 别名 / 描述 / worker 工具集 / worker 系统提示模板
  - 主 agent 派发时按「技能名」选 worker —— 新增一个场景 = 往注册表加一条，
    编排引擎一行不改（开放-封闭原则，也是 week13 技能清单思路的延续）。

内置两个技能（双场景演示，证明编排与具体场景解耦）：
  - weather：城市天气调研（open-meteo 免费 API，零 key，urllib 零依赖）
  - file：文档加工（读取本地文件 + LLM 总结/翻译/提炼）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

# ── 项目根（samples 目录安全边界）──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # week15/
DATA_DIR = PROJECT_ROOT / "samples"


# ════════════════════════════════════════════════════════════════════════
# 技能 1：weather —— 城市天气调研
# ════════════════════════════════════════════════════════════════════════
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "week15-orchestrator"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def city_weather(city: str) -> str:
    """城市名 → 结构化天气简报（当前 + 未来 3 天）。
    失败返回 [ERROR] 文本，不抛异常（ReAct 兜底，让模型修正参数重试）。"""
    try:
        geo = _http_get_json(GEO_URL + "?" + urllib.parse.urlencode(
            {"name": city, "count": 1, "language": "zh", "format": "json"}))
        hits = geo.get("results") or []
        if not hits:
            return f"[ERROR] 未找到城市「{city}」，请确认城市名拼写"
        loc = hits[0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc.get("name", city) + loc.get("admin1", "")
        fc = _http_get_json(FORECAST_URL + "?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": 3, "timezone": "auto"}))
        cur = fc["current"]
        cur_wmo = WEATHER_CODE_MAP.get(cur["weather_code"], f"码{cur['weather_code']}")
        lines = [
            f"【{city}】天气简报（{loc.get('country', '')} {name}）",
            f"- 当前：{cur_wmo}，{cur['temperature_2m']}°C，"
            f"湿度 {cur['relative_humidity_2m']}%，风速 {cur['wind_speed_10m']} km/h",
        ]
        for d, wmo, tmax, tmin in zip(
                fc["daily"]["time"], fc["daily"]["weather_code"],
                fc["daily"]["temperature_2m_max"], fc["daily"]["temperature_2m_min"]):
            lines.append(f"- {d}：{WEATHER_CODE_MAP.get(wmo, f'码{wmo}')}，"
                         f"{tmin}~{tmax}°C")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"[ERROR] 天气查询失败: {type(e).__name__}: {str(e)[:120]}"


WEATHER_PROMPT = """你是「城市天气调研员」。你的任务：查询指定城市的天气并输出结构化简报。

可用工具：
{tools_desc}

输出格式（每轮一次）：
Thought: 推理
Action: city_weather
Action Input: 城市名

拿到 Observation 后，直接输出：
Final Answer: 一份简洁的天气简报（天气现象、温度、湿度、风力、未来3天趋势），
并附一句实用建议（如是否需要带伞/增减衣物）。不要编造 Observation 之外的数据。"""


# ════════════════════════════════════════════════════════════════════════
# 技能 2：file —— 文档加工（读取本地文件，交给 LLM 总结/翻译/提炼）
# ════════════════════════════════════════════════════════════════════════
def _safe_path(raw: str) -> Path:
    """路径安全边界：只允许访问 week15/samples 内的文件（防目录穿越）。"""
    p = (DATA_DIR / raw.strip()).resolve()
    if not p.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"路径越界（只允许访问 {DATA_DIR.name}/ 内文件）: {raw}")
    return p


def read_file(spec: str) -> str:
    """读取 samples 目录内的文件。spec 示例: 'notes_rag.md' 或 'notes_rag.md:1-40'"""
    raw, _, rng = spec.strip().partition(":")
    p = _safe_path(raw)
    if not p.exists():
        return f"[ERROR] 文件不存在: {raw}（可用 list_files 查看可用文件）"
    text = p.read_text(encoding="utf-8")
    if rng:
        try:
            start, end = (int(x) for x in rng.split("-"))
            lines = text.splitlines()[start - 1:end]
            text = "\n".join(lines)
        except ValueError:
            return f"[ERROR] 行范围格式错误: {rng}，应为 开始-结束"
    return f"文件 {raw}（共 {len(text.splitlines())} 行）:\n{text[:3000]}"


def list_files(_: str = "") -> str:
    """列出 samples 目录下可用文件。"""
    files = sorted(p.name for p in DATA_DIR.glob("*.md"))
    if not files:
        return "[ERROR] samples 目录为空"
    return "可用文件:\n" + "\n".join(f"- {f}" for f in files)


FILE_PROMPT = """你是「文档加工员」。你的任务：按用户要求加工 samples 目录下的文档
（总结、翻译、提炼要点、改写等），加工前先用 read_file 读取文件内容。

可用工具：
{tools_desc}

输出格式（每轮一次）：
Thought: 推理
Action: read_file
Action Input: 文件名

拿到文件内容后，若信息足够则直接输出：
Final Answer: 加工结果（必须忠实于文件内容，不得编造原文没有的信息）。"""


# ════════════════════════════════════════════════════════════════════════
# 注册表：技能名 → {desc, tools, prompt}
# ════════════════════════════════════════════════════════════════════════
SKILL_REGISTRY: dict[str, dict] = {
    "weather": {
        "name": "weather",
        "aliases": ["天气", "天气调研"],
        "desc": "城市天气调研。任务参数=城市名，如「北京」",
        "tools": {"city_weather": (city_weather,
                                   "查询指定城市的天气。Action Input=城市名，如：北京")},
        "prompt": WEATHER_PROMPT,
        "max_steps": 3,
    },
    "file": {
        "name": "file",
        "aliases": ["文档", "文件"],
        "desc": "文档加工（总结/翻译/提炼要点）。任务参数=完整指令，如「用中文总结 samples/notes_rag.md 并提炼3个要点」",
        "tools": {"read_file": (read_file,
                                "读取 samples 目录内文件。Action Input=文件名，如：notes_rag.md"),
                  "list_files": (list_files,
                                 "列出 samples 目录可用文件。Action Input 可留空")},
        "prompt": FILE_PROMPT,
        "max_steps": 4,
    },
}


def resolve_skill(name: str) -> str | None:
    """技能名/别名 → 注册表 key；模糊前缀匹配兜底。"""
    n = name.strip().lower()
    if n in SKILL_REGISTRY:
        return n
    for key, spec in SKILL_REGISTRY.items():
        if n == key.lower() or n in [a.lower() for a in spec["aliases"]]:
            return key
    for key in SKILL_REGISTRY:          # 前缀模糊匹配
        if key.startswith(n) or n.startswith(key.lower()):
            return key
    return None


def list_skills_desc() -> str:
    """注册表 → 主 agent 可见的技能清单（L1 元数据视图，不加载实现）。"""
    lines = []
    for key, spec in SKILL_REGISTRY.items():
        alias = f"（别名：{'/'.join(spec['aliases'])}）" if spec["aliases"] else ""
        lines.append(f"- {key}{alias}：{spec['desc']}")
    return "\n".join(lines)
