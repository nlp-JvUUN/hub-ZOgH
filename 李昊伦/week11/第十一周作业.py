"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 拆成两个函数实现链式调用：geocode_city（城市名→经纬度）+ get_weather_by_coords（经纬度→天气）
     LLM 必须先调第一个拿到坐标，再用坐标调第二个——展示多轮循环的价值
  3. 保留 get_weather 便捷函数（内部串联两步），供非链式场景使用
  4. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import geocode_city, get_weather_by_coords, get_weather
  # 链式调用
  coords = geocode_city("宁德")
  weather = get_weather_by_coords(lat, lon, "宁德")
  # 或一步到位
  weather = get_weather("宁德")

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
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


def geocode_city(city: str) -> str:
    """
    第一步：城市名 → 经纬度坐标。链式调用的入口。

    Args:
        city: 城市中文名，如 "宁德"、"北京"。同名地名自动取行政级别更高的。

    Returns:
        JSON 字符串，包含 lat/lon/city_name/admin1/country，供 get_weather_by_coords 使用。
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

        if not results:
            return f"错误：未找到城市 '{city}'，请尝试其他写法（如加'市'后缀）"

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return json.dumps({
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "city_name": loc.get("name", city),
            "admin1": loc.get("admin1", ""),
            "country": loc.get("country", ""),
        }, ensure_ascii=False)


def get_weather_by_coords(lat: float, lon: float, city_name: str = "") -> str:
    """
    第二步：根据经纬度查询天气。需要先调 geocode_city 获取坐标。

    Args:
        lat: 纬度（从 geocode_city 返回）
        lon: 经度（从 geocode_city 返回）
        city_name: 可选，城市名（仅用于显示，默认用坐标）

    Returns:
        包含当前天气和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
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
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = city_name or f"{lat:.2f}°N, {lon:.2f}°E"

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
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
    便捷函数：一步到位查询天气（内部串联 geocode_city + get_weather_by_coords）。
    供非链式场景使用。
    """
    coords_str = geocode_city(city)
    if coords_str.startswith("错误"):
        return coords_str
    coords = json.loads(coords_str)
    return get_weather_by_coords(
        coords["lat"], coords["lon"],
        f"{coords.get('country', '')} {coords.get('admin1', '')} {coords.get('city_name', '')}".strip(),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
