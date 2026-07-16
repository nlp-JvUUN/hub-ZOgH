"""
run_mcp_loop.py — 方式二（循环调用版）：MCP Host + 多轮 ReAct 循环

与 run_mcp.py 的区别：
  单轮版：提问 → 工具调用（一轮）→ 最终回答
  循环版：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答（最多 max_rounds 轮）

教学重点：
  1. 多轮循环 + MCP 跨进程调用：每轮仍通过 session.call_tool() 跨进程执行
  2. 工具发现（list_tools）只做一次，后续轮次复用 openai_tools / tool_registry
  3. 与 run_function_call_loop.py 的循环逻辑完全一致，差异只在工具执行方式

使用方式：
  python mode_mcp/run_mcp_loop.py --demo
  python mode_mcp/run_mcp_loop.py -q "北京天气如何？如果下雨请再查上海"
"""

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

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


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


async def connect_all_servers(stack: AsyncExitStack):
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)

    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


# ── 多轮循环闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。你可以多次调用工具来获取信息，然后基于工具返回的结果"
    "决定是否需要调用更多工具，直到你收集到足够信息再给出最终回答。"
    "回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。你可以根据工具返回结果决定下一步操作。"
)


async def run_loop(client, model: str, question: str,
                  tool_registry: dict, openai_tools: list[dict],
                  max_rounds: int = 5, verbose: bool = True) -> dict:
    """
    多轮 ReAct 循环：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答。
    工具通过 MCP call_tool 跨进程执行。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        if verbose:
            print(f"\n  [round {rounds}] 模型思考中...")

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 最终回答（共 {rounds} 轮，{elapsed:.1f}s）")
            return {
                "answer": answer,
                "tool_calls": tool_call_log,
                "elapsed": elapsed,
                "rounds": rounds,
            }

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": rounds})
            if verbose:
                print(f"  → [mcp] {name}({args})")

            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if verbose:
        print(f"\n  [round {max_rounds}] 达到最大轮次，强制结束循环")
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=openai_tools, tool_choice="auto",
    )
    msg = resp.choices[0].message
    answer = msg.content or ""
    elapsed = time.time() - t0
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "rounds": max_rounds,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "北京天气如何？如果下雨请再查上海的天气做对比。",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比北京、上海、广州三座城市的天气。",
    "比亚迪2023年营收是多少？",
]


async def main_async(provider: str, question: str | None, demo: bool,
                    max_rounds: int, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP Loop] provider={provider} model={model} max_rounds={max_rounds}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)
        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run_loop(
                client, model, q, tool_registry, openai_tools,
                max_rounds=max_rounds, verbose=verbose and not as_json,
            )
            result["question"] = q
            results.append(result)
            if not as_json:
                print(f"\n共调用 {len(result['tool_calls'])} 次工具，{result['rounds']} 轮循环")
                print("最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP（多轮循环版）")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--loop", type=int, default=5, help="最大循环轮次（默认5）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(
        args.provider, args.question, args.demo,
        max_rounds=args.loop, verbose=not args.quiet, as_json=args.json,
    ))


if __name__ == "__main__":
    main()
