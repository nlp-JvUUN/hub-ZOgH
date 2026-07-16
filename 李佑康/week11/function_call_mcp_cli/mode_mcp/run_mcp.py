"""
run_mcp.py — MCP 方式：天气查询（循环交互）

使用方式：
  python mode_mcp/run_mcp.py
  python mode_mcp/run_mcp.py --provider dashscope

交互流程：
  用户输入城市名 → LLM 通过 MCP 调用 get_weather → 返回天气报告 → 循环
  输入 exit / quit 退出

依赖：
  pip install mcp openai httpx
  环境变量：DEEPSEEK_API_KEY（默认 LLM）或 DASHSCOPE_API_KEY（备选）
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOKE_API_KEY", ""),
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


# ── Server 配置 ────────────────────────────────────────────────────────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


# ── 连接 Server ────────────────────────────────────────────────────────────

async def connect_servers(stack: AsyncExitStack):
    """连接 weather MCP Server，返回 (tool_registry, openai_tools)。"""
    print("正在连接 MCP Server...\n", file=sys.stderr)
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


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气查询助手。用户会告诉你城市名，请调用 get_weather 工具查询天气，"
    "然后用自然语言总结返回的天气报告。本回合你可以一次调用多个工具。"
)


async def run_once(client, model: str, question: str,
                   tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> str:
    """单轮闭环：提问 → MCP tool_call → 路由到 Server 执行 → 回填 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model=model, messages=messages, tools=openai_tools, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  → [mcp] {name}({args})")

            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

            if verbose:
                preview = (result or "")[:120].replace("\n", " ")
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

    return msg.content or ""


# ── 循环交互入口 ───────────────────────────────────────────────────────────

async def main_async(provider: str):
    client, model = build_client(provider)
    print(f"天气查询助手（MCP）| provider={provider} model={model}")
    print("输入城市名查询天气，输入 exit / quit 退出\n")

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_servers(stack)

        while True:
            try:
                question = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("再见！")
                break

            answer = await run_once(client, model, question, tool_registry, openai_tools)
            print(f"\n助手：{answer}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="天气查询 — MCP 方式")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    args = parser.parse_args()
    asyncio.run(main_async(args.provider))


if __name__ == "__main__":
    main()
