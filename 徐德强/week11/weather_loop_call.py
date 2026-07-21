"""
weather_loop_call.py - 循环调用两个天气工具

改造目标：
  原来 get_weather(city) 在一个函数里完成：
    1. 城市名 -> 经纬度
    2. 经纬度 -> 天气

  现在改成循环调用：
    第 1 轮调用 get_location(city)
    第 2 轮调用 get_weather_by_location(latitude, longitude, name)
    拿到天气结果后结束循环

示例：
  python weather_loop_call.py --city 宁德
  python weather_loop_call.py --cities 宁德 北京 上海
  python weather_loop_call.py --cities 宁德,北京,上海
"""

import argparse
from typing import Any

import httpx

from get_location import get_location
from get_weather_by_location import get_weather_by_location


def call_tool(tool_name: str, tool_args: dict[str, Any]) -> Any:
    """
    统一工具执行入口。

    循环调度器只决定下一步调用哪个工具，真正的工具执行都收口到这里。
    """
    if tool_name == "get_location":
        return get_location(tool_args["city"])

    if tool_name == "get_weather_by_location":
        return get_weather_by_location(
            latitude=tool_args["latitude"],
            longitude=tool_args["longitude"],
            name=tool_args.get("name", "指定位置"),
        )

    raise ValueError(f"未知工具：{tool_name}")


def run_weather_loop(city: str, max_rounds: int = 5, verbose: bool = True) -> str:
    """
    用循环方式完成天气查询。

    Args:
        city: 城市名称
        max_rounds: 最大工具调用轮数，避免异常情况下无限循环
        verbose: 是否打印每轮调用过程

    Returns:
        最终天气回答或错误信息
    """
    state: dict[str, Any] = {
        "city": city,
        "location": None,
        "weather": None,
    }

    for round_index in range(1, max_rounds + 1):
        if state["location"] is None:
            tool_name = "get_location"
            tool_args = {"city": state["city"]}
        elif state["weather"] is None:
            location = state["location"]
            tool_name = "get_weather_by_location"
            tool_args = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "name": f"{location['country']} {location['admin1']} {location['city']}".strip(),
            }
        else:
            return state["weather"]

        if verbose:
            print(f"[第 {round_index} 轮] 调用工具：{tool_name}({tool_args})")

        try:
            tool_result = call_tool(tool_name, tool_args)
        except httpx.RequestError as exc:
            return f"工具调用失败：{exc}"

        if tool_name == "get_location":
            if tool_result is None:
                return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"
            state["location"] = tool_result
            if verbose:
                print(f"[第 {round_index} 轮] 工具结果：{tool_result}")
            continue

        if tool_name == "get_weather_by_location":
            state["weather"] = tool_result
            if verbose:
                print(f"[第 {round_index} 轮] 工具结果：天气查询完成")
            continue

    return f"超过最大工具调用轮数 {max_rounds}，仍未得到最终天气结果"


def normalize_cities(city: str | None, cities: list[str] | None) -> list[str]:
    raw_items = cities if cities else ([city] if city else [])
    result: list[str] = []
    for item in raw_items:
        for part in item.replace("，", ",").split(","):
            name = part.strip()
            if name:
                result.append(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="循环调用两个工具完成天气查询")
    parser.add_argument("--city", help="单个城市中文名，如 宁德")
    parser.add_argument("--cities", nargs="+", help="多个城市中文名，如 宁德 北京 上海；也支持 宁德,北京,上海")
    parser.add_argument("--max-rounds", type=int, default=5, help="最大循环轮数")
    parser.add_argument("--quiet", action="store_true", help="只输出最终结果")
    args = parser.parse_args()

    city_list = normalize_cities(args.city, args.cities)
    if not city_list:
        parser.error("请使用 --city 指定单个城市，或使用 --cities 指定多个城市")

    for index, city in enumerate(city_list, 1):
        if len(city_list) > 1:
            print(f"\n{'=' * 60}")
            print(f"城市 {index}/{len(city_list)}：{city}")
            print(f"{'=' * 60}")

        result = run_weather_loop(
            city=city,
            max_rounds=args.max_rounds,
            verbose=not args.quiet,
        )
        print("\n最终结果：")
        print(result)


if __name__ == "__main__":
    main()
