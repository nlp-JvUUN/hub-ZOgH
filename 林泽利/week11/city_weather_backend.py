"""
city_weather_backend.py — 城市、城市天气查询（三种方式共享的业务逻辑）


教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.city_weather_backend import get_weather, get_city_location
  print(get_city_location("宁德")) # 获取城市经纬度
  print(get_weather("宁德"))    # 获取城市天气

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx
from typing import Dict, List, Union

# 接口地址（适配Open-Meteo官方接口，规避robots限制，正常代码调用可用）
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


def get_city_location(city: str) -> Union[str, Dict]:
    """
    【独立方法】仅查询城市地理信息（经纬度、省份、国家、行政等级）
    :param city: 城市名称（支持中文）
    :return: 成功返回城市信息字典，失败返回错误字符串
    """
    def _geocode(name: str):
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

    # 首次查询城市
    results = _geocode(city)
    # 判定是否为低级行政点位（村庄、小镇）
    is_low_admin = all(
        str(r.get("feature_code", "")).startswith("PPL")
        and not str(r.get("feature_code", "")).startswith("PPLA")
        for r in results
    ) if results else True
    has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))

    # 无行政后缀且匹配低级点位，追加市字重查
    if is_low_admin and not has_suffix:
        retry_results = _geocode(city + "市")
        if retry_results:
            results = retry_results

    if not results:
        return f"未找到城市 '{city}'，请尝试其他写法（如补充市/县后缀）"

    # 优先级排序：行政级别高 & 有人口数据的城市优先
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    best_loc = max(results, key=_rank)
    # 精简返回核心地理信息
    return {
        "city": best_loc.get("name", city),
        "admin1": best_loc.get("admin1", ""),
        "country": best_loc.get("country", ""),
        "latitude": best_loc["latitude"],
        "longitude": best_loc["longitude"]
    }


def get_weather_by_lat_lon(lat: float, lon: float) -> Union[str, Dict]:
    """
    【独立方法】通过经纬度查询天气数据（当前天气+未来3天预报）
    :param lat: 纬度
    :param lon: 经度
    :return: 成功返回天气完整字典，失败返回错误字符串
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        return weather_resp.json()
    except httpx.RequestError as e:
        return f"天气数据获取失败：{str(e)}"


def format_weather_msg(loc_info: Dict, weather_data: Dict) -> str:
    """
    【独立方法】格式化地理+天气数据为可读中文文本（供LLM直接消费）
    :param loc_info: 城市地理信息字典
    :param weather_data: 天气原始接口数据
    :return: 格式化后的天气报告字符串
    """
    cur = weather_data["current"]
    daily = weather_data["daily"]

    # 天气状态映射
    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"未知天气(代码{cur['weather_code']})")
    location_str = f"{loc_info['country']} {loc_info['admin1']} {loc_info['city']}".strip()

    # 拼接输出文本
    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{loc_info['latitude']:.2f}°N, {loc_info['longitude']:.2f}°E",
        "",
        f"当前天气：{weather_desc}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]

    # 循环拼接三日预报
    for i in range(3):
        day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "未知天气")
        lines.append(
            f"  {daily['time'][i]}：{day_desc}，"
            f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )

    return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    【聚合方法】兼容原有用法：输入城市名，直接返回完整天气报告
    :param city: 城市名称
    :return: 格式化天气报告/错误提示
    """
    # 1. 获取城市地理信息
    loc_result = get_city_location(city)
    if isinstance(loc_result, str):
        return loc_result

    # 2. 通过经纬度获取天气
    weather_result = get_weather_by_lat_lon(
        lat=loc_result["latitude"],
        lon=loc_result["longitude"]
    )
    if isinstance(weather_result, str):
        return weather_result

    # 3. 格式化输出
    return format_weather_msg(loc_result, weather_result)

def batch_get_city_location(city_list: List[str]) -> List[Dict]:
    """
    【循环批量查询】支持传入城市列表，批量返回各城市地理信息
    :param city_list: 城市名称列表，如["宁德", "北京", "上海"]
    :return: 对应城市的地理信息
    """
    for city in city_list:
        result = get_city_location(city)
        print(f"===== {city} =====")
        print(result)
        print()
    

def batch_get_weather(city_list: List[str]) -> List[str]:
    """
    【循环批量查询】支持传入城市列表，批量返回各城市天气报告
    :param city_list: 城市名称列表，如["宁德", "北京", "上海"]
    :return: 对应城市的天气报告结果列表
    """
    result_list = []
    for city in city_list:
        result = get_weather(city)
        result_list.append(f"===== {city} =====" + "\n" + result)
    return result_list


# 测试入口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="天气查询工具（支持单城市/批量查询）")
    parser.add_argument("--city", type=str, help="单个查询城市名")
    parser.add_argument("--cities", nargs="+", help="批量查询城市列表（空格分隔）")

    args = parser.parse_args()

    # 单个城市查询
    if args.city:
        print("\n========== 单个城市经纬度查询 ==========\n")
        print(get_city_location(args.city))

        print("\n========== 单个城市天气查询 ==========\n")
        print(get_weather(args.city))

    # 批量循环查询
    if args.cities:
        print("\n========== 批量经纬度查询结果 ==========\n")
        batch_get_city_location(args.cities)
        
        print("\n========== 批量天气查询结果 ==========\n")
        batch_results = batch_get_weather(args.cities)
        for res in batch_results:
            print(res + "\n")
