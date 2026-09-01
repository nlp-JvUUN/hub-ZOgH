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
    5. get_comfort_index(temp,hum,wind) 温湿度风速 → 体感舒适度（下游衍生工具）

设计要点（与单函数版本的本质区别）：
  - 单一职责：每个工具只回答一类问题，输出可作为其它工具的输入
  - 结构化输出：geocode 返回 JSON 字符串，下游工具直接消费字段，无需文本解析
  - 显式错误协议：所有失败都返回以 [ERROR] 开头的文本（含错误码+修正建议），
    不抛异常 —— 这样 Agent Loop 才能把错误回填给 LLM，让模型在下一轮自我修正
    （这是"循环"相对"单轮"的关键价值之一，见 agent_loop.py）
  - 下游衍生工具：get_comfort_index 的输入（温度/湿度/风速）来自
    get_current_weather 的返回文本，由 LLM（或模拟决策器）提取后作为参数传入 ——
    演示"模型在循环里充当数据搬运工"的链式模式

数据源：Open-Meteo（Geocoding / Forecast / Air Quality 三个 API，均免费无 Key）。
运行：python weather_tools.py  （自带端到端自测）

依赖：pip install httpx
"""

import json
import re

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


def get_comfort_index(temperature: float, humidity: float, wind_speed: float) -> str:
    """
    工具 5（下游衍生）：体感舒适度。温度/湿度/风速 → 体感温度估算 + 舒适度等级 + 建议。

    设计要点：本工具不查任何 API，是纯"衍生计算"工具 —— 它的输入
    （temperature/humidity/wind_speed）来自 get_current_weather 的返回文本，
    由 LLM 在循环中提取后作为参数传入（schema 的 description 里写明了提取规则）。
    这演示了 Agent Loop 的另一种链式形态：工具 A 的输出 → 模型加工 → 工具 B 消费。

    估算规则（教学用简化模型，非气象学标准）：
      - 高温（≥26°C）：湿度越大越闷热，体感 ≈ 温度 + 闷热加成
      - 低温（≤10°C）：风速越大越冷，体感 ≈ 温度 − 风寒折扣
      - 其余温度区间：体感 ≈ 实际温度
    """
    try:
        t, h, w = float(temperature), float(humidity), float(wind_speed)
    except (TypeError, ValueError):
        return (f"[ERROR][PARAM] temperature/humidity/wind_speed 必须是数字，"
                f"收到 temperature={temperature!r}, humidity={humidity!r}, wind_speed={wind_speed!r}。"
                f"请从 get_current_weather 的返回文本中提取这些数值。")

    if t >= 26:
        feels = t + max(0.0, (h - 40) / 100.0) * 3.0      # 闷热加成
    elif t <= 10:
        feels = t - (w / 10.0) * 2.0                        # 风寒折扣
    else:
        feels = t

    if feels >= 35:
        level, advice = "酷热", "避免长时间户外活动，注意防暑补水"
    elif feels >= 30:
        level, advice = "炎热", "注意防晒补水，午后尽量减少外出"
    elif feels >= 26:
        level, advice = "闷热", "体感偏热，注意通风降温"
    elif feels >= 18:
        level, advice = "舒适", "体感舒适，适合户外活动"
    elif feels >= 10:
        level, advice = "偏凉", "建议加一件薄外套"
    elif feels >= 0:
        level, advice = "寒冷", "注意保暖，戴好围巾手套"
    else:
        level, advice = "严寒", "注意防寒防滑，尽量减少外出"

    return (
        f"体感温度约 {feels:.1f}°C（舒适度等级：{level}）\n"
        f"依据：气温 {t:.1f}°C / 湿度 {h:.0f}% / 风速 {w:.1f} km/h\n"
        f"建议：{advice}"
    )


# 工具名 → 函数 的注册表（供 agent_loop.py 的 TOOL_DISPATCH 复用，保持单一来源）
TOOLS = {
    "geocode": geocode,
    "get_current_weather": get_current_weather,
    "get_daily_forecast": get_daily_forecast,
    "get_air_quality": get_air_quality,
    "get_comfort_index": get_comfort_index,
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

    print("\n== 4) 下游衍生：从 get_current_weather 提取参数 → get_comfort_index ==")
    cur = get_current_weather(lat, lon)
    print(cur)
    m_t = re.search(r"温度：(-?\d+\.?\d*)°C", cur)
    m_h = re.search(r"相对湿度：(-?\d+\.?\d*)%", cur)
    m_w = re.search(r"风速：(-?\d+\.?\d*) km/h", cur)
    if m_t and m_h and m_w:
        print(get_comfort_index(float(m_t.group(1)), float(m_h.group(1)), float(m_w.group(1))))

    print("\n== 5) 错误协议（循环自愈的触发源）==")
    print("geocode('绿野仙踪') →", geocode("绿野仙踪")[:60], "...")
    print("get_current_weather(999, 0) →", get_current_weather(999, 0)[:60], "...")
    print("get_comfort_index('abc', 50, 10) →", get_comfort_index("abc", 50, 10)[:60], "...")
