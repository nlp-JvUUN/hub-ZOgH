"""
weather_tools.py — 天气 Agent 的工具层（纯业务逻辑，与 LLM 完全解耦）

作业背景：
  课件 mode_function_call 里的 get_weather(city) 把"城市名→经纬度"和"经纬度→天气"
  揉在一个函数里，单轮闭环一次调用就完成。本作业把它拆成 4 个原子工具，
  由 Agent Loop 里的 LLM 自主决定调用顺序、次数与组合方式：

    1. geocode(city)               城市名 → 结构化位置（经纬度）
    2. get_current_weather(lat,lon) 坐标 → 当前天气
    3. get_daily_forecast(lat,lon)  坐标 → 未来 N 天预报
    4. get_air_quality(lat,lon)     坐标 → 空气质量

设计要点（与单函数版本的本质区别）：
  - 单一职责：每个工具只回答一类问题，输出可作为其它工具的输入
  - 结构化输出：geocode 返回 JSON 字符串，下游工具直接消费字段，无需文本解析
  - 显式错误协议：所有失败都返回以 [ERROR] 开头的文本（含错误码+修正建议），
    不抛异常 —— 这样 Agent Loop 才能把错误回填给 LLM，让模型在下一轮自我修正
    （这是"循环"相对"单轮"的关键价值之一，见 agent_loop.py）

数据源：Open-Meteo（Geocoding / Forecast / Air Quality 三个 API，均免费无 Key）。
运行：python weather_tools.py  （自带端到端自测）

依赖：pip install httpx
"""

import json

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo 天气代码 → 中文描述（沿用课件映射，保持同一数据口径）
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}

# US AQI 指数 → 中文等级（Open-Meteo 的 us_aqi 采用 US EPA 标准）
AQI_LEVELS = [
    (0, 50, "优"),
    (51, 100, "良"),
    (101, 150, "轻度污染"),
    (151, 200, "中度污染"),
    (201, 300, "重度污染"),
    (301, 9999, "严重污染"),
]


def _check_coords(latitude: float, longitude: float) -> str | None:
    """坐标合法性校验。合法返回 None，非法返回 [ERROR] 文本（供工具直接返回）。"""
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return f"[ERROR][PARAM] 坐标必须是数字，收到 latitude={latitude!r}, longitude={longitude!r}"
    if not (-90 <= lat <= 90):
        return (f"[ERROR][PARAM] 纬度 {lat} 超出合法范围 [-90, 90]。"
                f"注意：纬度是南北方向的度数，经度是东西方向的度数，两者不要写反。")
    if not (-180 <= lon <= 180):
        return f"[ERROR][PARAM] 经度 {lon} 超出合法范围 [-180, 180]。"
    return None


def _lookup_city(client: httpx.Client, name: str) -> list:
    """调用 Open-Meteo Geocoding 并返回候选列表。"""
    resp = client.get(GEOCODING_URL, params={
        "name": name, "count": 10, "language": "zh", "format": "json",
    })
    resp.raise_for_status()
    return resp.json().get("results") or []


def _pick_city(client: httpx.Client, city: str) -> dict | None:
    """
    城市名 → 最优地理条目（含同名消歧，逻辑继承自课件 weather_backend）：
      1. 按原名查；若结果全是低级行政点（feature_code 纯 PPL）且用户没带
         "市/县/区/镇"后缀，则用 city+"市" 重查一次并优先采用；
      2. 候选按（行政级别 PPLA/ADM > 普通点，人口数）排序取最优。
    找不到返回 None。
    """
    results = _lookup_city(client, city)
    is_low_admin = all(
        str(r.get("feature_code", "")).startswith("PPL")
        and not str(r.get("feature_code", "")).startswith("PPLA")
        for r in results
    ) if results else True
    has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
    if is_low_admin and not has_suffix:
        retry = _lookup_city(client, city + "市")
        if retry:
            results = retry
    if not results:
        return None

    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        return (admin_priority, r.get("population") or 0)

    return max(results, key=_rank)


def geocode(city: str) -> str:
    """
    工具 1：地理编码。城市名（中文）→ 结构化位置 JSON。

    返回 JSON 字符串（下游工具可直接消费字段）：
      {"name": "宁德", "country": "中国", "admin1": "福建省",
       "latitude": 26.66, "longitude": 119.52}
    找不到时返回 [ERROR][NOT_FOUND] + 修正建议，由 LLM 决定是换写法重试还是如实告知用户。
    """
    city = (city or "").strip()
    if not city:
        return "[ERROR][PARAM] 城市名不能为空，请提供中文城市名，如 '北京'、'宁德'。"
    with httpx.Client(timeout=10.0) as client:
        loc = _pick_city(client, city)
    if loc is None:
        return (f"[ERROR][NOT_FOUND] 未找到城市 '{city}'。"
                f"建议检查拼写，或补充行政区后缀（如 '北京市'），或换一个更常见的城市名。")
    return json.dumps({
        "name": loc.get("name", city),
        "country": loc.get("country", ""),
        "admin1": loc.get("admin1", ""),
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
    }, ensure_ascii=False)


def _fetch_forecast(latitude: float, longitude: float) -> dict:
    """内部共用：请求未来 3 天预报原始数据（当前+逐日）。"""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(FORECAST_URL, params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        resp.raise_for_status()
        return resp.json()


def get_current_weather(latitude: float, longitude: float) -> str:
    """
    工具 2：当前天气。坐标 → 天气状况、温度、湿度、风速。

    坐标越界 / 非数字 → [ERROR][PARAM]（Agent Loop 借此触发模型自我修正）。
    """
    err = _check_coords(latitude, longitude)
    if err:
        return err
    try:
        data = _fetch_forecast(latitude, longitude)
    except httpx.RequestError as e:
        return f"[ERROR][NETWORK] 天气数据获取失败：{e}"
    cur = data["current"]
    desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    return (
        f"当前天气：{desc}\n"
        f"温度：{cur['temperature_2m']}°C\n"
        f"相对湿度：{cur['relative_humidity_2m']}%\n"
        f"风速：{cur['wind_speed_10m']} km/h"
    )


def get_daily_forecast(latitude: float, longitude: float, days: int = 3) -> str:
    """
    工具 3：未来预报。坐标 → 未来 days 天（1~7）逐日天气。

    与工具 2 拆开的理由：用户只问"今天"时不必拉取逐日预报；
    用户问"未来几天"时也不必重复拿当前实况 —— 由 LLM 按需选择。
    """
    err = _check_coords(latitude, longitude)
    if err:
        return err
    try:
        days = int(days)
    except (TypeError, ValueError):
        return f"[ERROR][PARAM] days 必须是整数（1~7），收到 {days!r}。"
    days = max(1, min(7, days))
    try:
        data = _fetch_forecast(latitude, longitude)
    except httpx.RequestError as e:
        return f"[ERROR][NETWORK] 预报数据获取失败：{e}"
    daily = data["daily"]
    lines = [f"未来{days}天预报："]
    for i in range(min(days, len(daily["time"]))):
        desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "未知")
        lines.append(
            f"  {daily['time'][i]}：{desc}，"
            f"最高 {daily['temperature_2m_max'][i]}°C / 最低 {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )
    return "\n".join(lines)


def get_air_quality(latitude: float, longitude: float) -> str:
    """
    工具 4：空气质量。坐标 → US AQI 指数、等级、PM2.5/PM10/O₃/NO₂。

    独立的第三个数据源（Open-Meteo Air Quality API），与天气 API 并列 ——
    让 LLM 可以在同一轮循环里"并行"调 get_current_weather + get_air_quality。
    """
    err = _check_coords(latitude, longitude)
    if err:
        return err
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(AIR_QUALITY_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",
                "timezone": "Asia/Shanghai",
            })
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError as e:
        return f"[ERROR][NETWORK] 空气质量数据获取失败：{e}"

    cur = data["current"]
    aqi = cur["us_aqi"]
    level = "未知"
    for lo, hi, name in AQI_LEVELS:
        if lo <= aqi <= hi:
            level = name
            break
    return (
        f"空气质量指数 AQI：{aqi}（{level}）\n"
        f"PM2.5：{cur['pm2_5']} µg/m³\n"
        f"PM10：{cur['pm10']} µg/m³\n"
        f"臭氧 O₃：{cur['ozone']} µg/m³\n"
        f"二氧化氮 NO₂：{cur['nitrogen_dioxide']} µg/m³"
    )


# 工具名 → 函数 的注册表（供 agent_loop.py 的 TOOL_DISPATCH 复用，保持单一来源）
TOOLS = {
    "geocode": geocode,
    "get_current_weather": get_current_weather,
    "get_daily_forecast": get_daily_forecast,
    "get_air_quality": get_air_quality,
}


if __name__ == "__main__":
    # 端到端自测（不依赖任何 LLM / API Key）：
    # 用真实 API 手动演示"链式"与"错误协议"，验证工具层可用。
    print("== 1) geocode('宁德') ==")
    info = geocode("宁德")
    print(info)
    loc = json.loads(info)
    lat, lon = loc["latitude"], loc["longitude"]

    print("\n== 2) 链式：用 geocode 结果查 get_current_weather ==")
    print(get_current_weather(lat, lon))

    print("\n== 3) 独立：get_daily_forecast 与 get_air_quality ==")
    print(get_daily_forecast(lat, lon, days=2))
    print(get_air_quality(lat, lon))

    print("\n== 4) 错误协议（循环自愈的触发源）==")
    print("geocode('亚特兰蒂斯') →", geocode("亚特兰蒂斯")[:60], "...")
    print("get_current_weather(999, 0) →", get_current_weather(999, 0)[:60], "...")
