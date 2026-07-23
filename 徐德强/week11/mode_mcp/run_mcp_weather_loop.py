"""
run_mcp_weather_loop.py - MCP 多轮循环调用天气工具

流程：
  1. Host 启动 mcp_weather_server.py
  2. Host 通过 list_tools 自动发现 get_location / get_weather_by_location
  3. LLM 多轮选择工具，Host 通过 MCP call_tool 执行

运行：
  python run_mcp_weather_loop.py --question "查询宁德天气"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


MODE_DIR = Path(__file__).parent
BASE_DIR = MODE_DIR.parent

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


async def connect_weather_server(stack: AsyncExitStack) -> tuple[ClientSession, list[dict[str, Any]]]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MODE_DIR / "mcp_weather_server.py")],
        env={**os.environ},
    )
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    tools_result = await session.list_tools()
    openai_tools = []
    for tool in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return session, openai_tools


async def run(
    client: OpenAI,
    model: str,
    question: str,
    session: ClientSession,
    openai_tools: list[dict[str, Any]],
    max_rounds: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "你是天气助手。查询天气必须分两步循环调用 MCP 工具："
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
            tools=openai_tools,
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
                print(f"[第 {round_index} 轮] MCP 调用：{name}({args})")

            call_result = await session.call_tool(name, args)
            result = "\n".join(block.text for block in call_result.content if hasattr(block, "text"))
            tool_result_log.append({
                "round": round_index,
                "name": name,
                "result_preview": str(result)[:300],
            })
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return {
        "answer": f"超过最大循环轮数 {max_rounds}，仍未得到最终回答",
        "tool_calls": tool_call_log,
        "tool_results": tool_result_log,
        "elapsed": time.time() - started,
    }


async def run_many(
    client: OpenAI,
    model: str,
    question: str,
    cities: list[str],
    session: ClientSession,
    openai_tools: list[dict[str, Any]],
    max_rounds: int,
    verbose: bool,
) -> dict[str, Any]:
    started = time.time()
    results = []
    all_tool_calls = []
    all_tool_results = []

    for city_index, city in enumerate(cities, 1):
        if verbose and len(cities) > 1:
            print(f"\n{'=' * 60}\n城市 {city_index}/{len(cities)}：{city}\n{'=' * 60}")
        result = await run(
            client,
            model,
            f"查询{city}天气",
            session,
            openai_tools,
            max_rounds=max_rounds,
            verbose=verbose,
        )
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


async def main_async(args: argparse.Namespace) -> None:
    client, model = build_client(args.provider)
    async with AsyncExitStack() as stack:
        session, openai_tools = await connect_weather_server(stack)
        cities = normalize_cities(args.question, args.cities)
        if len(cities) == 1:
            result = await run(
                client,
                model,
                f"查询{cities[0]}天气",
                session,
                openai_tools,
                max_rounds=args.max_rounds,
                verbose=not args.quiet,
            )
        else:
            result = await run_many(
                client,
                model,
                args.question,
                cities,
                session,
                openai_tools,
                max_rounds=args.max_rounds,
                verbose=not args.quiet,
            )
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
        return

    print("\n最终回答：")
    print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP 多轮循环天气查询")
    parser.add_argument("--question", "-q", default="查询宁德天气")
    parser.add_argument("--cities", nargs="+", help="多个城市，如 深圳 北京 上海；也支持 深圳,北京,上海")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON，供对比脚本解析")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
