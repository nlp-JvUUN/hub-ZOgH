"""
weather_backend_new.py — 天气查询后端（拆分版，支持链式调用）

教学重点：
  1. 将原 get_weather 拆分为两个独立 function：
       - get_geocoding(city)  : 城市名 → 经纬度（可单独调用）
       - get_weather(city)    : 城市名 → 天气（内部链式调用 get_geocoding）
  2. 链式调用：get_weather 内部先调用 get_geocoding 获取经纬度，再查天气
  3. 两个 function 均可被 LLM function-calling 独立调用，便于演示工具链
  4. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend_new import get_geocoding, get_weather

  # 单独查询经纬度
  print(get_geocoding("宁德"))

  # 直接查询天气（内部自动链式调用 get_geocoding）
  print(get_weather("宁德"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

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


def _geocode(client: httpx.Client, name: str):
    """内部辅助：调用 Geocoding API 返回候选列表。"""
    resp = client.get(GEOCODING_URL, params={
        "name": name, "count": 10, "language": "zh", "format": "json",
    })
    resp.raise_for_status()
    return resp.json().get("results") or []


def _pick_best_location(results, city: str):
    """内部辅助：从候选中选出行政级别最高、人口最多的地点。"""
    # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
    # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
    # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
    # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
    is_low_admin = all(
        str(r.get("feature_code", "")).startswith("PPL")
        and not str(r.get("feature_code", "")).startswith("PPLA")
        for r in results
    ) if results else True
    has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
    if is_low_admin and not has_suffix:
        with httpx.Client(timeout=10.0) as c2:
            retry = _geocode(c2, city + "市")
        if retry:
            results = retry

    if not results:
        return None

    # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
    # 其次取有人口数据的，避免落到同名小村庄
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    return max(results, key=_rank)


def get_geocoding(city: str) -> str:
    """
    查询指定城市的经纬度坐标及行政区划信息。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含城市名、国家、省/州、纬度、经度、行政级别、人口的文字描述；
        若未找到城市则返回提示字符串
    """
    with httpx.Client(timeout=10.0) as client:
        results = _geocode(client, city)
        loc = _pick_best_location(results, city)

    if not loc:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德'改'宁德市'）"

    lat = loc["latitude"]
    lon = loc["longitude"]
    city_name = loc.get("name", city)
    country = loc.get("country", "")
    admin1 = loc.get("admin1", "")       # 省/州级行政区
    admin2 = loc.get("admin2", "")       # 地市级行政区
    feature_code = loc.get("feature_code", "")
    population = loc.get("population", 0)

    lines = [
        f"【{country} {admin1} {city_name}】地理信息",
        f"  纬度：{lat:.4f}°N",
        f"  经度：{lon:.4f}°E",
        f"  行政区划代码：{feature_code}",
        f"  省/州：{admin1}",
        f"  地市：{admin2}",
        f"  人口：{population}",
    ]
    return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    内部链式调用：先调用 get_geocoding 获取经纬度，再用经纬度查询天气。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    # ===== Step 1：链式调用 get_geocoding 获取经纬度 =====
    with httpx.Client(timeout=10.0) as client:
        results = _geocode(client, city)
        loc = _pick_best_location(results, city)

    if not loc:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德'改'宁德市'）"

    lat = loc["latitude"]
    lon = loc["longitude"]
    city_name = loc.get("name", city)
    country = loc.get("country", "")
    admin1 = loc.get("admin1", "")  # 省/州级行政区

    # ===== Step 2：用经纬度查询天气 =====
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

    # ===== Step 3：格式化输出 =====
    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    location_str = f"{country} {admin1} {city_name}".strip()

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--mode", choices=["geo", "weather", "both"], default="both",
        help="geo=仅查经纬度；weather=仅查天气；both=两者都查（默认）",
    )
    args = parser.parse_args()

    if args.mode in ("geo", "both"):
        print("===== 经纬度查询 =====")
        print(get_geocoding(args.city))
        print()
    if args.mode in ("weather", "both"):
        print("===== 天气查询 =====")
        print(get_weather(args.city))
