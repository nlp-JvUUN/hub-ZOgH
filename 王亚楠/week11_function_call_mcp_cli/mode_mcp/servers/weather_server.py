"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. get_weather 拆分为三个独立工具：geocode / get_weather_by_coords / get_weather
  3. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.weather_backend import (  # noqa: E402
    geocode as _geocode,
    get_weather as _get_weather,
    get_weather_by_coords as _get_weather_by_coords,
)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def geocode(city: str) -> str:
    """
    查询指定城市的经纬度坐标。

    Args:
        city: 城市中文名，如 '宁德'、'北京'、'上海'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含城市名、国家、省份、经纬度的文字描述。
    """
    return _geocode(city)


@mcp.tool()
def get_weather_by_coords(lat: float, lon: float) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        lat: 纬度，十进制，如 26.67。
        lon: 经度，十进制，如 119.55。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather_by_coords(lat, lon)


@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报（便捷工具：自动 geocode → 天气查询）。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather(city)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
