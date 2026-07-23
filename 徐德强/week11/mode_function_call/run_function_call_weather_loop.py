"""
run_function_call_weather_loop.py - Function Call 多轮循环调用天气工具

流程：
  1. LLM 调用 get_location(city)，得到 latitude / longitude
  2. LLM 根据上一步结果继续调用 get_weather_by_location(latitude, longitude, name)
  3. LLM 汇总最终天气回答

运行：
  python run_function_call_weather_loop.py --question "查询宁德天气"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from get_location import get_location
from get_weather_by_location import get_weather_by_location


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
            "name": "get_location",
            "description": "根据城市中文名查询经纬度。查询天气前必须先调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 宁德、北京"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_location",
            "description": "根据经纬度查询当前天气和未来3天预报。必须使用 get_location 返回的经纬度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度"},
                    "longitude": {"type": "number", "description": "经度"},
                    "name": {"type": "string", "description": "位置名称，用于输出展示"},
                },
                "required": ["latitude", "longitude"],
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


def call_tool(name: str, args: dict[str, Any]) -> str:
    if name == "get_location":
        result = get_location(args["city"])
        return json.dumps(result, ensure_ascii=False)

    if name == "get_weather_by_location":
        return get_weather_by_location(
            latitude=args["latitude"],
            longitude=args["longitude"],
            name=args.get("name", "指定位置"),
        )

    return f"未知工具：{name}"


def create_chat_completion(client: OpenAI, **kwargs: Any):
    last_error = None
    for _ in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise last_error


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
    return result or [question]


def run(client: OpenAI, model: str, question: str, max_rounds: int = 5, verbose: bool = True) -> dict[str, Any]:
    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "你是天气助手。查询天气必须分两步循环调用工具："
                "第一步调用 get_location 获取经纬度；"
                "第二步根据经纬度调用 get_weather_by_location；"
                "拿到天气结果后再回答用户。"
            ),
        },
        {"role": "user", "content": question},
    ]
    tool_call_log = []
    tool_result_log = []
    started = time.time()

    for round_index in range(1, max_rounds + 1):
        resp = create_chat_completion(
            client,
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
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
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"round": round_index, "name": name, "args": args})
            if verbose:
                print(f"[第 {round_index} 轮] Function Call 调用：{name}({args})")

            result = call_tool(name, args)
            tool_result_log.append({
                "round": round_index,
                "name": name,
                "result_preview": str(result)[:300],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return {
        "answer": f"超过最大循环轮数 {max_rounds}，仍未得到最终回答",
        "tool_calls": tool_call_log,
        "tool_results": tool_result_log,
        "elapsed": time.time() - started,
    }


def run_many(client: OpenAI, model: str, question: str, cities: list[str], max_rounds: int, verbose: bool) -> dict[str, Any]:
    started = time.time()
    results = []
    all_tool_calls = []
    all_tool_results = []

    for city_index, city in enumerate(cities, 1):
        if verbose and len(cities) > 1:
            print(f"\n{'=' * 60}\n城市 {city_index}/{len(cities)}：{city}\n{'=' * 60}")
        result = run(client, model, f"查询{city}天气", max_rounds=max_rounds, verbose=verbose)
        for call in result.get("tool_calls", []):
            call["city"] = city
            all_tool_calls.append(call)
        for item in result.get("tool_results", []):
            item["city"] = city
            all_tool_results.append(item)
        results.append({"city": city, **result})

    answer = "\n\n".join(f"## {item['city']}\n{item['answer']}" for item in results)
    return {
        "answer": answer,
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "city_results": results,
        "elapsed": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Function Call 多轮循环天气查询")
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
        result = run(client, model, f"查询{cities[0]}天气", args.max_rounds, verbose=not (args.quiet or args.json))
    else:
        result = run_many(client, model, args.question, cities, args.max_rounds, verbose=not (args.quiet or args.json))
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
        return

    print("\n最终回答：")
    print(result["answer"])


if __name__ == "__main__":
    main()
