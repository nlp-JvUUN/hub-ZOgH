"""
weather_server.py — 天气查询 MCP Server
底层不再提供 get_weather，手动串联地理编码+天气请求
"""
import sys
import httpx
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mcp.server.fastmcp import FastMCP
# 只导入两个底层工具
from src.weather_backend import _geocode_city, _fetch_weather_by_latlng, MAX_RETRY, RETRY_DELAY, WEATHER_CODE_MAP

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)
mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。
    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的城市。
    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    # 分步调用底层函数
    location_info = _geocode_city(city)
    if location_info is None:
        return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

    lat = location_info["latitude"]
    lon = location_info["longitude"]
    city_name = location_info.get("name", city)
    country = location_info.get("country", "")
    admin1 = location_info.get("admin1", "")

    # 循环重试天气接口
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
            return f"天气接口连续{MAX_RETRY}次请求失败，错误信息：{err_msg}，请稍后重试"

    # MCP层手动格式化输出
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
    return "\n".join(lines)

if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")