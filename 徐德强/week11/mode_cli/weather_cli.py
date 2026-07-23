"""
weather_cli.py - 天气命令行工具

运行：
  python weather_cli.py location --city 宁德
  python weather_cli.py weather --latitude 26.66167 --longitude 119.52278 --name 宁德
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from get_location import get_location
from get_weather_by_location import get_weather_by_location


def main() -> None:
    parser = argparse.ArgumentParser(description="天气 CLI 工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_location = sub.add_parser("location", help="根据城市名查询经纬度")
    p_location.add_argument("--city", required=True)

    p_weather = sub.add_parser("weather", help="根据经纬度查询天气")
    p_weather.add_argument("--latitude", type=float, required=True)
    p_weather.add_argument("--longitude", type=float, required=True)
    p_weather.add_argument("--name", default="指定位置")

    args = parser.parse_args()
    if args.cmd == "location":
        print(json.dumps(get_location(args.city), ensure_ascii=False))
    elif args.cmd == "weather":
        print(get_weather_by_location(args.latitude, args.longitude, args.name))


if __name__ == "__main__":
    main()
