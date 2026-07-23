# weather_server.py — 改进版，提供两个独立工具
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP
from src.tool_backend import get_coordinates as _get_coordinates
from src.tool_backend import get_weather as _get_weather

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("weather-server")

mcp = FastMCP("weather-server")

# ---------- 工具 1：获取坐标 ----------
@mcp.tool()
def get_coordinates(city: str) -> str:
    """
    根据城市中文名获取经纬度（返回 JSON 字符串，含 latitude 和 longitude）。
    """
    try:
        result = _get_coordinates(city)  # 后端返回 JSON 字符串
        # 验证 JSON 合法性
        json.loads(result)
        return result
    except Exception as e:
        logger.error(f"坐标查询失败：{e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ---------- 工具 2：查询天气 ----------
@mcp.tool()
def get_weather(latitude: float, longitude: float) -> str:
    """
    根据经纬度获取当前天气及未来3天预报（返回文本描述）。
    """
    try:
        result = _get_weather(latitude, longitude)
        # 截断防止溢出（但一般不会太长）
        if isinstance(result, str) and len(result) > 2000:
            result = result[:2000] + "...(截断)"
        return result
    except Exception as e:
        logger.error(f"天气查询失败：{e}", exc_info=True)
        return f"天气查询失败：{e}"

if __name__ == "__main__":
    logger.info("Weather MCP Server 启动（stdio）")
    mcp.run(transport="stdio")