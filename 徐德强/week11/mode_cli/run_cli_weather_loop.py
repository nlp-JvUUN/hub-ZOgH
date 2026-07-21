"""
run_cli_weather_loop.py - CLI 多轮循环调用天气工具

流程：
  1. LLM 调用 run_cli(command="location", args={"city": "宁德"})
  2. Host 执行 python weather_cli.py location --city 宁德
  3. LLM 根据 stdout 继续调用 run_cli(command="weather", args={...})
  4. Host 执行 python weather_cli.py weather --latitude ... --longitude ...

运行：
  python run_cli_weather_loop.py --question "查询宁德天气"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


MODE_DIR = Path(__file__).parent
BASE_DIR = MODE_DIR.parent
PY = sys.executable

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的天气命令行工具。"
                "查询天气必须先 command=location 获取经纬度，再 command=weather 查询天气。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["location", "weather"],
                        "description": "location 根据城市查经纬度；weather 根据经纬度查天气",
                    },
                    "args": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "location 命令使用，城市中文名，如 宁德",
                            },
                            "latitude": {
                                "type": "number",
                                "description": "weather 命令使用，纬度",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "weather 命令使用，经度",
                            },
                            "name": {
                                "type": "string",
                                "description": "weather 命令使用，位置名称，如 宁德市",
                            },
                        },
                        "description": (
                            "location 参数：{city}; "
                            "weather 参数：{latitude, longitude, name?}"
                        ),
                    },
                },
                "required": ["command", "args"],
            },
        },
    },
]


def build_client(provider: str) -> tuple[OpenAI, str]:
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def create_chat_completion(client: OpenAI, **kwargs: Any):
    last_error = None
    for _ in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise last_error


def run_cli(command: str, args: dict[str, Any]) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"命令参数不是合法 JSON：{args}"

    if command == "location":
        argv = [PY, str(MODE_DIR / "weather_cli.py"), "location", "--city", str(args["city"])]
    elif command == "weather":
        argv = [
            PY,
            str(MODE_DIR / "weather_cli.py"),
            "weather",
            "--latitude",
            str(args["latitude"]),
            "--longitude",
            str(args["longitude"]),
            "--name",
            str(args.get("name", "指定位置")),
        ]
    else:
        return f"未知命令：{command}"

    try:
        proc = subprocess.run(
            argv,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return "命令执行超时"

    if proc.returncode != 0:
        return f"命令执行失败，退出码 {proc.returncode}：{proc.stderr[-500:]}"

    output = proc.stdout
    if command == "location":
        try:
            location = json.loads(output)
        except json.JSONDecodeError:
            return output
        if location is None:
            return f"未找到城市：{args['city']}"
        return (
            "经纬度查询成功："
            f"{json.dumps(location, ensure_ascii=False)}\n"
            "下一步必须继续调用 run_cli，参数为："
            f"command='weather', args={{'latitude': {location['latitude']}, "
            f"'longitude': {location['longitude']}, 'name': '{location['city']}'}}"
        )

    return output


def extract_city(question: str) -> str:
    city = question.strip()
    for word in (
        "我要去", "我想去", "我去", "去",
        "给我", "帮我", "请",
        "看一下", "看看", "查一下", "查询", "查",
        "什么", "一下", "天气", "的",
    ):
        city = city.replace(word, "")
    return city.strip() or question.strip()


def normalize_cities(question: str, cities: list[str] | None = None) -> list[str]:
    if cities:
        raw_text = ",".join(cities)
    else:
        raw_text = question
        for word in (
            "我要去", "我想去", "我去", "去",
            "给我", "帮我", "请",
            "看一下", "看看", "查一下", "查询", "查",
            "什么", "一下", "天气", "的",
        ):
            raw_text = raw_text.replace(word, "")

    for sep in ("，", "、", "和", "以及", "；", ";", " "):
        raw_text = raw_text.replace(sep, ",")
    result = [item.strip() for item in raw_text.split(",") if item.strip()]
    return result or [extract_city(question)]


def run_one_city(city: str, verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    tool_call_log = []
    tool_result_log = []

    location_args = {"city": city}
    if verbose:
        print(f"[第 1 轮] CLI 调用：run_cli(location, {location_args})")
    location_result = run_cli("location", location_args)
    tool_call_log.append({
        "round": 1,
        "name": "run_cli",
        "args": {"command": "location", "args": location_args},
    })
    tool_result_log.append({
        "round": 1,
        "command": "location",
        "result_preview": str(location_result)[:300],
    })

    marker = "经纬度查询成功："
    first_line = str(location_result).splitlines()[0] if str(location_result).splitlines() else ""
    if not first_line.startswith(marker):
        return {
            "answer": str(location_result),
            "tool_calls": tool_call_log,
            "tool_results": tool_result_log,
            "elapsed": time.time() - started,
        }

    location = json.loads(first_line[len(marker):])
    weather_args = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "name": location.get("city", city),
    }
    if verbose:
        print(f"[第 2 轮] CLI 调用：run_cli(weather, {weather_args})")
    weather_result = run_cli("weather", weather_args)
    tool_call_log.append({
        "round": 2,
        "name": "run_cli",
        "args": {"command": "weather", "args": weather_args},
    })
    tool_result_log.append({
        "round": 2,
        "command": "weather",
        "result_preview": str(weather_result)[:300],
    })

    return {
        "answer": str(weather_result),
        "tool_calls": tool_call_log,
        "tool_results": tool_result_log,
        "elapsed": time.time() - started,
    }


def run(client: OpenAI, model: str, question: str, max_rounds: int = 5, verbose: bool = True) -> dict[str, Any]:
    cities = normalize_cities(question)
    if len(cities) == 1:
        return run_one_city(cities[0], verbose=verbose)

    started = time.time()
    results = []
    all_tool_calls = []
    all_tool_results = []
    for city_index, city in enumerate(cities, 1):
        if verbose:
            print(f"\n{'=' * 60}\n城市 {city_index}/{len(cities)}：{city}\n{'=' * 60}")
        result = run_one_city(city, verbose=verbose)
        for call in result.get("tool_calls", []):
            call["city"] = city
            all_tool_calls.append(call)
        for item in result.get("tool_results", []):
            item["city"] = city
            all_tool_results.append(item)
        results.append({"city": city, **result})

    return {
        "answer": "\n\n".join(f"## {item['city']}\n{item['answer']}" for item in results),
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "city_results": results,
        "elapsed": time.time() - started,
    }


def run_with_cities(cities: list[str], verbose: bool = True) -> dict[str, Any]:
    started = time.time()
    results = []
    all_tool_calls = []
    all_tool_results = []
    for city_index, city in enumerate(cities, 1):
        if verbose and len(cities) > 1:
            print(f"\n{'=' * 60}\n城市 {city_index}/{len(cities)}：{city}\n{'=' * 60}")
        result = run_one_city(city, verbose=verbose)
        for call in result.get("tool_calls", []):
            call["city"] = city
            all_tool_calls.append(call)
        for item in result.get("tool_results", []):
            item["city"] = city
            all_tool_results.append(item)
        results.append({"city": city, **result})

    return {
        "answer": "\n\n".join(f"## {item['city']}\n{item['answer']}" for item in results),
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "city_results": results,
        "elapsed": time.time() - started,
    }

    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "你是天气助手。只能通过 run_cli 调用命令行工具。"
                "查询天气必须分两步循环调用：先 location 获取经纬度，再 weather 查询天气。"
                "拿到 location 的经纬度结果后，不要重复调用 location，必须立刻用返回的 latitude/longitude 调用 weather。"
                "拿到 weather 的天气结果后再回答用户。"
            ),
        },
        {"role": "user", "content": question},
    ]
    tool_call_log = []
    tool_result_log = []
    last_location = None
    started = time.time()

    for round_index in range(1, max_rounds + 1):
        try:
            resp = create_chat_completion(
                client,
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
        except Exception as exc:
            return {
                "answer": f"LLM 请求失败：{type(exc).__name__}: {exc}",
                "tool_calls": tool_call_log,
                "elapsed": time.time() - started,
            }
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {
                "answer": msg.content or "",
                "tool_calls": tool_call_log,
                "tool_results": tool_result_log,
                "elapsed": time.time() - started,
            }

        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            command = args["command"]
            command_args = args.get("args", {})
            if isinstance(command_args, str):
                try:
                    command_args = json.loads(command_args)
                    args["args"] = command_args
                except json.JSONDecodeError:
                    pass
            tool_call_log.append({"round": round_index, "name": "run_cli", "args": args})
            if verbose:
                print(f"[第 {round_index} 轮] CLI 调用：run_cli({command}, {command_args})")

            result = run_cli(command, command_args)
            tool_result_log.append({
                "round": round_index,
                "command": command,
                "result_preview": str(result)[:300],
            })

            if command == "location":
                try:
                    marker = "经纬度查询成功："
                    line = str(result).splitlines()[0]
                    if line.startswith(marker):
                        last_location = json.loads(line[len(marker):])
                except (IndexError, json.JSONDecodeError, TypeError):
                    last_location = None

            if command == "location" and last_location is not None and round_index >= 2:
                weather_args = {
                    "latitude": last_location["latitude"],
                    "longitude": last_location["longitude"],
                    "name": last_location.get("city", "指定位置"),
                }
                weather_result = run_cli("weather", weather_args)
                tool_call_log.append({
                    "round": round_index,
                    "name": "run_cli",
                    "args": {"command": "weather", "args": weather_args},
                })
                tool_result_log.append({
                    "round": round_index,
                    "command": "weather",
                    "result_preview": str(weather_result)[:300],
                })
                return {
                    "answer": str(weather_result),
                    "tool_calls": tool_call_log,
                    "tool_results": tool_result_log,
                    "elapsed": time.time() - started,
                }

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

            if command == "weather" and "天气报告" in str(result):
                return {
                    "answer": str(result),
                    "tool_calls": tool_call_log,
                    "tool_results": tool_result_log,
                    "elapsed": time.time() - started,
                }

    return {
        "answer": f"超过最大循环轮数 {max_rounds}，仍未得到最终回答",
        "tool_calls": tool_call_log,
        "tool_results": tool_result_log,
        "elapsed": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI 多轮循环天气查询")
    parser.add_argument("--question", "-q", default="查询宁德天气")
    parser.add_argument("--cities", nargs="+", help="多个城市，如 深圳 北京 上海；也支持 深圳,北京,上海")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON，供对比脚本解析")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    cities = normalize_cities(args.question, args.cities)
    if len(cities) == 1:
        result = run_one_city(cities[0], verbose=not (args.quiet or args.json))
    else:
        result = run_with_cities(cities, verbose=not (args.quiet or args.json))
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
        return

    print("\n最终回答：")
    print(result["answer"])


if __name__ == "__main__":
    main()
