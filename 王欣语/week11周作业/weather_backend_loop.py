"""
weather_backend_loop.py — 天气查询工具（循环调用版本）

在 weather_backend.py 基础上改造：
  1. 支持多城市批量查询（循环调用）
  2. 支持交互式循环输入（用户可连续查询多个城市）
  3. 支持导出结果到文件

使用方式：
  # 方式1：命令行传多个城市
  python weather_backend_loop.py --cities 北京 上海 深圳 宁德

  # 方式2：交互式循环输入（输入 q 退出）
  python weather_backend_loop.py --interactive

  # 方式3：导出到文件
  python weather_backend_loop.py --cities 北京 上海 --output weather_report.txt

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，以便导入 src 模块
sys.path.insert(0, str(Path(__file__).parent))
from weather_backend import get_weather


def batch_query(cities: list[str]) -> dict[str, str]:
    """
    批量查询多个城市天气，循环调用 get_weather。

    Args:
        cities: 城市名称列表

    Returns:
        dict: {城市名: 天气报告}
    """
    results = {}
    print(f"开始批量查询 {len(cities)} 个城市...\n")

    for i, city in enumerate(cities, 1):
        print(f"[{i}/{len(cities)}] 正在查询 {city} 的天气...")
        try:
            report = get_weather(city)
            results[city] = report
            print(f"  ✓ {city} 查询成功")
        except Exception as e:
            results[city] = f"查询失败：{e}"
            print(f"  ✗ {city} 查询失败：{e}")
        print()  # 空行分隔

    return results


def interactive_loop():
    """
    交互式循环：用户持续输入城市名，输入 q/quit/exit 退出。
    """
    print("=" * 60)
    print("天气查询工具（循环调用模式）")
    print("=" * 60)
    print("提示：输入城市名查询天气，输入 q/quit/exit 退出\n")

    history = []  # 记录查询历史

    while True:
        try:
            user_input = input("请输入城市名：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit", "退出"):
            print(f"\n共查询了 {len(history)} 个城市，再见！")
            break

        # 支持一次输入多个城市（用空格或逗号分隔）
        cities = [c.strip() for c in user_input.replace(",", " ").split() if c.strip()]

        for city in cities:
            print(f"\n正在查询 {city}...")
            report = get_weather(city)
            print(report)
            print("-" * 40)
            history.append(city)


def print_summary(results: dict[str, str]):
    """
    打印批量查询结果汇总。
    """
    print("\n" + "=" * 60)
    print("查询结果汇总")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if not r.startswith("查询失败"))
    fail_count = len(results) - success_count

    print(f"总计：{len(results)} 个城市 | 成功：{success_count} | 失败：{fail_count}\n")

    for city, report in results.items():
        if report.startswith("查询失败"):
            print(f"✗ {city}: {report}")
        else:
            # 只打印第一行（城市名）和当前温度
            lines = report.split("\n")
            if len(lines) >= 5:
                print(f"✓ {lines[0]}")
                print(f"  {lines[4]}")  # 当前天气行
            else:
                print(f"✓ {city}: 查询成功")
        print()


def save_to_file(results: dict[str, str], filepath: str):
    """
    将查询结果保存到文件。
    """
    path = Path(filepath)
    lines = [
        "天气查询报告",
        "=" * 60,
        f"查询城市数：{len(results)}",
        "",
    ]

    for city, report in results.items():
        lines.append(report)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存到：{path.absolute()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="天气查询工具（循环调用版本）"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        help="要查询的城市列表，例如：北京 上海 深圳",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="进入交互式循环查询模式",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="将结果保存到指定文件路径",
    )

    args = parser.parse_args()

    # 交互式模式
    if args.interactive:
        interactive_loop()
        return

    # 批量查询模式
    if args.cities:
        results = batch_query(args.cities)
        print_summary(results)

        if args.output:
            save_to_file(results, args.output)
        return

    # 默认：进入交互式模式
    print("未指定城市，进入交互式查询模式...\n")
    interactive_loop()


if __name__ == "__main__":
    main()
