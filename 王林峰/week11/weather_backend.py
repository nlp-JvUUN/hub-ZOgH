"""
weather_backend.py — 天气查询底层工具（无统一封装入口）
教学重点：
  1. 拆分两大独立函数，各自创建独立 httpx.Client，完全隔离网络连接
  2. 不再提供统一 get_weather 封装方法，由上层业务自行串联地理编码+天气查询+重试循环
  3. _geocode_city：城市名解析坐标；_fetch_weather_by_latlng：经纬度拉取天气原始JSON
依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""
import httpx
import time

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# 重试配置：最多重试3次，间隔1秒
MAX_RETRY = 3
RETRY_DELAY = 1
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


def _geocode_city(city_name: str) -> dict | None:
    """
    Step1 独立工具：城市名称转地理坐标
    内部新建独立 httpx.Client，不与天气请求共享连接
    返回匹配度最高的城市地理字典，无匹配返回 None
    """
    with httpx.Client(timeout=10.0) as geo_client:
        def _raw_geocode(name: str):
            resp = geo_client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _raw_geocode(city_name)
        # 处理村镇歧义，无市后缀自动补充重试
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city_name.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry_results = _raw_geocode(city_name + "市")
            if retry_results:
                results = retry_results

        if not results:
            return None

        # 按行政级别、人口优先级排序
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        return max(results, key=_rank)


def _fetch_weather_by_latlng(lat: float, lon: float):
    """
    Step2 独立工具：根据经纬度获取天气原始JSON数据
    内部新建独立 httpx.Client，与地理编码完全隔离
    请求异常直接抛出 RequestError，由上层实现循环重试逻辑
    """
    with httpx.Client(timeout=10.0) as weather_client:
        weather_resp = weather_client.get(WEATHER_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        weather_resp.raise_for_status()
        return weather_resp.json()


# 【删除原有统一入口 get_weather 函数】
# 格式化文本逻辑迁移至各上层调用处，不再下沉到backend