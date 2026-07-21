#!/usr/bin/env python3
"""
weather_cli — 基于 Open-Meteo 的经纬度 & 天气查询命令行工具

教学重点：
  1. CLI 是“工具暴露”的最原始形式——直接调用后端函数，不依赖 LLM。
  2. 子命令设计（coords / weather）清晰分离两种功能。
  3. 统一错误处理、日志、JSON 输出，便于与其他模式（Function Call / MCP）对比。
  4. 可作为 console_script 打包，也可直接用 `python -m` 运行。

使用方式（直接运行）：
  python -m mode_cli.cli.weather_cli coords --city 宁德
  python -m mode_cli.cli.weather_cli weather --city 成都
  python -m mode_cli.cli.weather_cli weather --lat 30.57 --lon 104.07

安装后全局调用：
  pip install -e .
  weather-cli coords --city 北京 --json
  weather-cli weather --city 上海 --verbose

依赖：
  pip install httpx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tool_backend import get_coordinates, get_weather

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weather_cli")

# ---------- 工具函数 ----------
def truncate_text(text: str, max_len: int = 2000) -> str:
    """截断过长的输出"""
    if len(text) > max_len:
        return text[:max_len] + "\n... (截断)"
    return text

def output_result(data, args_json):
    """统一输出：若 --json 则输出 JSON，否则打印文本"""
    if args_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        if isinstance(data, dict):
            # 如果是坐标，格式化输出
            if "latitude" in data and "longitude" in data:
                print(f"经度: {data['longitude']}, 纬度: {data['latitude']}")
            else:
                print(json.dumps(data, ensure_ascii=False))
        else:
            print(data)

# ---------- 子命令实现 ----------
def cmd_coords(args):
    """查询城市坐标"""
    try:
        result = get_coordinates(args.city)
        # get_coordinates 返回 JSON 字符串，解析为 dict
        coord = json.loads(result)
        output_result(coord, args.json)
    except Exception as e:
        logger.error(f"坐标查询失败: {e}", exc_info=True)
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

def cmd_weather(args):
    """查询天气（支持城市名或经纬度）"""
    try:
        if args.city:
            # 先通过城市名获取坐标
            coord_str = get_coordinates(args.city)
            coord = json.loads(coord_str)
            lat, lon = coord["latitude"], coord["longitude"]
        elif args.lat is not None and args.lon is not None:
            lat, lon = args.lat, args.lon
        else:
            raise ValueError("必须提供 --city 或同时提供 --lat 和 --lon")
        
        weather_text = get_weather(lat, lon)
        if args.json:
            # 尽量将天气文本解析为结构化数据（但后端返回的是文本，我们包装成字典）
            # 这里简单将文本按行拆分作为数组
            lines = weather_text.split('\n')
            output_data = {
                "latitude": lat,
                "longitude": lon,
                "weather": lines
            }
            output_result(output_data, args.json)
        else:
            print(truncate_text(weather_text))
    except Exception as e:
        logger.error(f"天气查询失败: {e}", exc_info=True)
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

# ---------- 主解析器 ----------
def main():
    parser = argparse.ArgumentParser(
        prog="weather-cli",
        description="查询城市坐标和天气（基于 Open-Meteo）"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # 子命令：coords
    p_coords = sub.add_parser("coords", help="查询城市经纬度")
    p_coords.add_argument("--city", required=True, help="城市中文名")

    # 子命令：weather
    p_weather = sub.add_parser("weather", help="查询天气")
    group = p_weather.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", help="城市中文名（优先使用）")
    group.add_argument("--lat", type=float, help="纬度")
    group.add_argument("--lon", type=float, help="经度（需与 --lat 同时使用）")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.cmd == "coords":
        cmd_coords(args)
    elif args.cmd == "weather":
        cmd_weather(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()