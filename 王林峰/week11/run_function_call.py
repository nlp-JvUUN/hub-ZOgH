# 头部导入修改
import httpx
import time
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_backend import search_annual_report, list_companies
# 替换为底层两个函数
from src.weather_backend import _geocode_city, _fetch_weather_by_latlng, MAX_RETRY, RETRY_DELAY, WEATHER_CODE_MAP

# TOOL_DISPATCH 仅保留年报工具，天气逻辑内联处理
TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    # get_weather 移除映射，在调用处手动实现串联逻辑
}

# run 函数内工具调用分支修改
if name == "get_weather":
    city = args.get("city", "")
    # 分步调用底层函数
    location_info = _geocode_city(city)
    if location_info is None:
        result = f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"
    else:
        lat = location_info["latitude"]
        lon = location_info["longitude"]
        city_name = location_info.get("name", city)
        country = location_info.get("country", "")
        admin1 = location_info.get("admin1", "")
        weather_data = None
        err_msg = ""
        for retry_times in range(MAX_RETRY):
            try:
                weather_data = _fetch_weather_by_latlng(lat, lon)
                break
            except httpx.RequestError as e:
                err_msg = str(e)
                if retry_times < MAX_RETRY - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                weather_data = None
        if weather_data is None:
            result = f"天气接口连续{MAX_RETRY}次请求失败，错误信息：{err_msg}，请稍后重试"
        else:
            cur = weather_data["current"]
            daily = weather_data["daily"]
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
            result = "\n".join(lines)