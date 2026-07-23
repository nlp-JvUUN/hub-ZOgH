"""
tool_backend.py 

提供获取经纬度和天气查询接口 （function_call、mcp、cli三种方式共享的业务逻辑）

"""


# tool_backend.py
import json
import httpx
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5),
       retry=retry_if_exception_type(httpx.RequestError))
@lru_cache(maxsize=128)
def get_coordinates(city: str) -> str:
    """根据城市名称获取经纬度（返回 JSON 字符串）"""
    with httpx.Client(timeout=10.0) as client:
        # 优先尝试带后缀的查询（避免裸名误定位）
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={"name": name, "count": 10, "language": "zh", "format": "json"})
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if not results and has_suffix:
            results = _geocode(city[:-1])
        if not results:
            return f"未找到城市 '{city}'"

        # 选择优先级最高的结果
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            population = r.get("population") or 0
            return (admin_priority, population)

        loc = max(results, key=_rank)
        return json.dumps({"latitude": loc["latitude"], "longitude": loc["longitude"]}, ensure_ascii=False)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=3))
def get_weather(latitude: float, longitude: float) -> str:
    """根据经纬度获取天气（返回格式化的文本）"""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(WEATHER_URL, params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        resp.raise_for_status()
        data = resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        lines = [
            f"【当前天气】{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "【未来3天】",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )
        return "\n".join(lines)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    latitude, longitude = get_coordinates(args.city)
    weather = get_weather(latitude, longitude)
    print(weather)