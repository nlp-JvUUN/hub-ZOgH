"""
main.py — fincli：天气查询命令行入口（作业精简版，仅含 weather 子命令）

把 src/weather_backend 封装成一条真实命令，通过 pyproject.toml 注册为 console_script：
  pip install -e .  →  fincli weather --city 北京

不想安装也可直接跑：
  python mode_cli/cli/main.py weather --city 北京
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weather_backend import get_weather  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — 天气查询命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")

    args = parser.parse_args()

    if args.cmd == "weather":
        print(get_weather(args.city))


if __name__ == "__main__":
    main()
