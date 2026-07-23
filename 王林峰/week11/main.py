"""
main.py — fincli：A股年报检索 + 天气查询 统一命令行入口
把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.rag_backend import search_annual_report, list_companies
# 改动：只导入两个底层函数，移除 get_weather
from src.weather_backend import _geocode_city, _fetch_weather_by_latlng, MAX_RETRY, RETRY_DELAY, WEATHER_CODE_MAP
import httpx
import time

def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    # fincli search
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True, help="检索财务术语，不含公司年份")
    p_search.add_argument("--stock-code", default=None)
    p_search.add_argument("--year", default=None)
    p_search.add_argument("--top-k", type=int, default=5)
    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录公司")
    # fincli weather
    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True)
    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "weather":
        city = args.city
        # 手动分步调用底层函数，替代原get_weather
        location_info = _geocode_city(city)
        if location_info is None:
            print(f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）")
            return

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
                print(f"天气接口连续{MAX_RETRY}次请求失败，错误信息：{err_msg}，请稍后重试")
                return
        # 上层手动格式化输出
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
        print("\n".join(lines))

if __name__ == "__main__":
    main()