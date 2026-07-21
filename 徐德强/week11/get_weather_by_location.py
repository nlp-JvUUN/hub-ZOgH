"""
get_weather_by_location.py - 根据经纬度查询天气

示例：
  python get_weather_by_location.py --latitude 26.67 --longitude 119.52 --name 宁德
"""

import argparse

import httpx


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


def get_weather_by_location(latitude: float, longitude: float, name: str = "指定位置") -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        latitude: 纬度
        longitude: 经度
        name: 位置名称，仅用于输出展示

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
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
        except httpx.RequestError as exc:
            return f"天气数据获取失败：{exc}"

    data = weather_resp.json()
    cur = data["current"]
    daily = data["daily"]

    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    lines = [
        f"【{name}】天气报告",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="根据经纬度查询天气")
    parser.add_argument("--latitude", type=float, required=True, help="纬度")
    parser.add_argument("--longitude", type=float, required=True, help="经度")
    parser.add_argument("--name", default="指定位置", help="位置名称，用于输出展示")
    args = parser.parse_args()

    print(get_weather_by_location(args.latitude, args.longitude, args.name))


if __name__ == "__main__":
    main()
