"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

核心设计：
  1. 纯业务逻辑层，与 rag_backend 平级，被三种方式复用
  2. 链式调用设计：拆成三个原子函数，用循环串联，为后续 Agent 扩展做准备
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import get_weather
  print(get_weather("北京"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def geocode_city(client: httpx.Client, city: str) -> dict:
    """
    原子函数1：城市名 → 经纬度坐标

    Args:
        client: httpx.Client 实例
        city:   城市名称，支持中文

    Returns:
        包含 lat, lon, city_name, country, admin1 的字典，或含 error 的字典
    """
    def _geocode(name: str):
        resp = client.get(GEOCODING_URL, params={
            "name": name, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        return resp.json().get("results") or []

    try:
        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return {"error": f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"}

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "city_name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }
    except Exception as e:
        return {"error": f"地理编码失败：{e}"}


def query_weather_by_coords(client: httpx.Client, lat: float, lon: float) -> dict:
    """
    原子函数2：经纬度 → 原始天气数据

    Args:
        client: httpx.Client 实例
        lat:    纬度
        lon:    经度

    Returns:
        包含 weather_data 的字典，或含 error 的字典
    """
    try:
        weather_resp = client.get(WEATHER_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        weather_resp.raise_for_status()
        return {"weather_data": weather_resp.json()}
    except Exception as e:
        return {"error": f"天气数据获取失败：{e}"}


def format_weather(city_name: str, country: str, admin1: str, weather_data: dict) -> str:
    """
    原子函数3：原始天气数据 → 格式化文本

    Args:
        city_name:    城市名称
        country:      国家
        admin1:       省级行政区
        weather_data: query_weather_by_coords 返回的原始数据

    Returns:
        格式化的天气报告文本
    """
    cur = weather_data["current"]
    daily = weather_data["daily"]

    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    location_str = f"{country} {admin1} {city_name}".strip()

    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{weather_data['latitude']:.2f}°N, {weather_data['longitude']:.2f}°E",
        "",
        f"当前天气：{weather_desc}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]
    for i in range(3):
        day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
        lines.append(
            f"  {daily['time'][i]}：{day_desc}，"
            f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )

    return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    采用 Agent 循环调用模式：
    while state has next step:
        tool = select_next_tool(state)
        result = execute_tool(tool, state)
        state = update_state(state, result)

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    def select_next_tool(state: dict) -> str | None:
        """根据当前状态选择下一个要执行的工具"""
        if "coords" not in state:
            return "geocode_city"
        elif "weather_data" not in state:
            return "query_weather_by_coords"
        elif "formatted" not in state:
            return "format_weather"
        return None

    def execute_tool(tool_name: str, state: dict, client: httpx.Client) -> dict:
        """执行指定工具"""
        if tool_name == "geocode_city":
            return geocode_city(client, state["city"])
        elif tool_name == "query_weather_by_coords":
            return query_weather_by_coords(client, state["coords"]["lat"], state["coords"]["lon"])
        elif tool_name == "format_weather":
            return {"result": format_weather(
                state["coords"]["city_name"],
                state["coords"]["country"],
                state["coords"]["admin1"],
                state["weather_data"],
            )}
        return {"error": f"未知工具：{tool_name}"}

    def update_state(state: dict, tool_name: str, result: dict) -> dict:
        """根据工具执行结果更新状态"""
        if "error" in result:
            state["error"] = result["error"]
            return state
        if tool_name == "geocode_city":
            state["coords"] = result
        elif tool_name == "query_weather_by_coords":
            state["weather_data"] = result["weather_data"]
        elif tool_name == "format_weather":
            state["formatted"] = result["result"]
        return state

    with httpx.Client(timeout=10.0) as client:
        state = {"city": city}

        while True:
            tool_name = select_next_tool(state)
            if tool_name is None or "error" in state:
                break

            result = execute_tool(tool_name, state, client)
            state = update_state(state, tool_name, result)

        if "error" in state:
            return state["error"]
        return state.get("formatted", "天气查询失败")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
