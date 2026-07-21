"""
mcp_weather_server.py - 天气工具 MCP Server

由 run_mcp_weather_loop.py 作为子进程启动，使用 stdio 通信。
"""

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from get_location import get_location as _get_location
from get_weather_by_location import get_weather_by_location as _get_weather_by_location


mcp = FastMCP("weather-loop-server")


@mcp.tool()
def get_location(city: str) -> str:
    """
    根据城市中文名查询经纬度。查询天气前必须先调用本工具。

    Args:
        city: 城市中文名，如 宁德、北京。
    """
    return json.dumps(_get_location(city), ensure_ascii=False)


@mcp.tool()
def get_weather_by_location(latitude: float, longitude: float, name: str = "指定位置") -> str:
    """
    根据经纬度查询当前天气和未来3天预报。

    Args:
        latitude: 纬度。
        longitude: 经度。
        name: 位置名称，用于输出展示。
    """
    return _get_weather_by_location(latitude, longitude, name)


if __name__ == "__main__":
    print("Weather Loop MCP Server 启动中...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
