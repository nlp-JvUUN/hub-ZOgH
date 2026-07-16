"""
weather_backend_v2.py — 天气查询后端 v2（经纬度查询与天气查询拆分为两个方法）

相对 weather_backend.py 的改动：
  原来 get_weather(city) 内部一次性做完"城市名 → 经纬度"和"经纬度 → 天气"两步。
  这里拆成两个独立的对外方法，分别对应两次 HTTP 请求，方便暴露成两个独立的
  Function Call 工具，由模型自己决定先查坐标、再用坐标查天气（多轮工具调用）：

    1. geocode_city(city)              城市名 → 经纬度（Geocoding API）
    2. get_weather_by_coords(lat, lon)  经纬度 → 当前天气 + 未来3天预报

  get_weather(city) 仍保留，作为两步的内部封装，供 CLI/直接调用等不需要
"""

import json
import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _geocode_lookup(city: str) -> list[dict]:
    """
    内部辅助：城市名 → 候选地点列表（原始 API 结果，未做排序/兜底）。

    中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
    而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
    策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
    且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

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

        return results


def geocode_city(city: str) -> str:
    """
    查询城市名对应的经纬度坐标（第一步：Geocoding）。
    """
    results = _geocode_lookup(city)
    if not results:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

    # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
    # 其次取有人口数据的，避免落到同名小村庄
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    loc = max(results, key=_rank)
    return json.dumps({
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "city_name": loc.get("name", city),
        "country": loc.get("country", ""),
        "admin1": loc.get("admin1", ""),  # 省/州级行政区
    }, ensure_ascii=False)


def get_weather_by_coords(latitude: float, longitude: float, location_name: str = "") -> str:
    """
    根据经纬度查询当前天气及未来3天预报（第二步：天气查询）。
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

    data = weather_resp.json()
    cur = data["current"]
    daily = data["daily"]

    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    location_str = location_name or f"{latitude:.2f}°N, {longitude:.2f}°E"

    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
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
    查询指定城市的当前天气及未来3天预报（geocode_city + get_weather_by_coords 的封装）。
    """
    geo_raw = geocode_city(city)
    try:
        geo = json.loads(geo_raw)
    except json.JSONDecodeError:
        return geo_raw  # geocode_city 返回的是错误提示字符串

    location_str = f"{geo.get('country', '')} {geo.get('admin1', '')} {geo.get('city_name', city)}".strip()
    return get_weather_by_coords(geo["latitude"], geo["longitude"], location_name=location_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
