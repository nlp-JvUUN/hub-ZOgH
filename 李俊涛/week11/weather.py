"""
weather_backend.py — 天气查询后端

对外提供三个函数：
  - geocode(city)       : 城市名 → 经纬度 + 行政区划信息（返回字典）
  - forecast(lat, lon)  : 经纬度 → 当前天气 + 3天预报（返回字典）
  - get_weather(city)   : 便捷组合函数，内部调 geocode → forecast，返回格式化文本

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx
from typing import Any, Dict, Optional

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


def _geocode_api(client: httpx.Client, name: str) -> list:
    """内部函数：调用 Geocoding API 并返回 results 列表（可能为空）"""
    try:
        resp = client.get(GEOCODING_URL, params={
            "name": name, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        return resp.json().get("results") or []
    except Exception:
        return []


def _pick_best(results: list) -> Dict[str, Any]:
    """从候选结果中选出行政区划级别最高、人口最多的一个"""
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    return max(results, key=_rank)


def geocode(city: str) -> Dict[str, Any]:
    """
    根据城市名称查询经纬度和行政区划信息。

    Args:
        city: 城市名称，支持中文，如 "宁德"、"北京"、"上海"

    Returns:
        {
            "lat": float,        # 纬度
            "lon": float,        # 经度
            "display_name": str, # 格式化后的地点全名
            "country": str,      # 国家
            "admin1": str        # 一级行政区（省/州）
        }
        或查询失败时返回 {"error": "错误描述"}
    """
    with httpx.Client(timeout=10.0) as client:
        results = _geocode_api(client, city)

        # 纠偏逻辑：若用户没带“市/县/区”后缀，且当前结果都是低级居住点，
        # 则尝试拼接“市”再查一次（避免“宁德”命中西藏村庄）
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry_results = _geocode_api(client, city + "市")
            if retry_results:
                results = retry_results

        if not results:
            return {"error": f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改为'宁德'）"}

        loc = _pick_best(results)
        display_name = " ".join(
            filter(None, [loc.get("country", ""), loc.get("admin1", ""), loc.get("name", city)])
        )
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "display_name": display_name,
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }


def forecast(lat: float, lon: float) -> Dict[str, Any]:
    """
    根据经纬度获取当前天气及未来3天预报。

    Args:
        lat: 纬度
        lon: 经度

    Returns:
        {
            "current": {
                "temperature_2m": float,
                "relative_humidity_2m": int,
                "wind_speed_10m": float,
                "weather_code": int,
                "weather_desc": str    # 中文描述
            },
            "daily": [
                {
                    "time": str,
                    "weather_code": int,
                    "weather_desc": str,
                    "temp_max": float,
                    "temp_min": float,
                    "precipitation_sum": float
                },
                ...  # 共3天
            ]
        }
        或查询失败时返回 {"error": "错误描述"}
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"天气数据获取失败：{e}"}

        data = resp.json()
        current = data["current"]
        daily = data["daily"]

        return {
            "current": {
                "temperature_2m": current["temperature_2m"],
                "relative_humidity_2m": current["relative_humidity_2m"],
                "wind_speed_10m": current["wind_speed_10m"],
                "weather_code": current["weather_code"],
                "weather_desc": WEATHER_CODE_MAP.get(current["weather_code"], f"代码{current['weather_code']}"),
            },
            "daily": [
                {
                    "time": daily["time"][i],
                    "weather_code": daily["weather_code"][i],
                    "weather_desc": WEATHER_CODE_MAP.get(daily["weather_code"][i], f"代码{daily['weather_code'][i]}"),
                    "temp_max": daily["temperature_2m_max"][i],
                    "temp_min": daily["temperature_2m_min"][i],
                    "precipitation_sum": daily["precipitation_sum"][i],
                }
                for i in range(3)
            ],
        }


def get_weather(city: str) -> str:
    """
    便捷组合函数：先查经纬度，再查天气，返回格式化文本（与原来完全兼容）。

    Args:
        city: 城市名称

    Returns:
        多行天气报告字符串，或错误提示
    """
    geo = geocode(city)
    if "error" in geo:
        return geo["error"]

    weather = forecast(geo["lat"], geo["lon"])
    if "error" in weather:
        return weather["error"]

    cur = weather["current"]
    daily = weather["daily"]

    lines = [
        f"【{geo['display_name']}】天气报告",
        f"坐标：{geo['lat']:.2f}°N, {geo['lon']:.2f}°E",
        "",
        f"当前天气：{cur['weather_desc']}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]
    for day in daily:
        lines.append(
            f"  {day['time']}：{day['weather_desc']}，"
            f"{day['temp_max']}°C / {day['temp_min']}°C，"
            f"降水 {day['precipitation_sum']} mm"
        )
    return "\n".join(lines)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
