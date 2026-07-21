"""
weather_agent.py — 方式二：MCP ReAct Agent（天气查询多轮循环）

教学重点：
  1. MCP 协议下的 ReAct 循环：工具通过 list_tools() 自动发现，执行走 session.call_tool()
  2. Host 端统一管理 Server 子进程生命周期（AsyncExitStack）
  3. 与 Function Call 版本的对比：工具定义从"手写 schema"变为"协议发现"，
     执行从"直接调函数"变为"跨进程 JSON-RPC"——但 ReAct 循环逻辑相同

使用方式：
  python mode_mcp/weather_agent.py -q "北京上海广州哪个最热？"
  python mode_mcp/weather_agent.py --demo

环境变量：
  DEEPSEEK_API_KEY（默认 LLM） / DASHSCOPE_API_KEY（备选 LLM）
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

# ═══════════════════════════════════════════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Server 配置（只连接 weather_server）
# ═══════════════════════════════════════════════════════════════════════════════

def build_server_config() -> StdioServerParameters:
    server_path = BASE_DIR / "mode_mcp" / "servers" / "weather_server.py"
    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env={**os.environ},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 连接 Server：建管道 → 握手 → 发现工具 → 转 OpenAI schema
# ═══════════════════════════════════════════════════════════════════════════════

async def connect_server(stack: AsyncExitStack):
    """连接 weather_server，返回 (session, openai_tools)。"""
    print("正在连接 MCP Weather Server...\n", file=sys.stderr)

    params = build_server_config()
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
                "parameters": (
                    tool.inputSchema
                    or {"type": "object", "properties": {}}
                ),
            },
        })

    names = ", ".join(t.name for t in tools_result.tools)
    print(f"  ✓ [weather] {names}", file=sys.stderr)
    print(f"\n共 {len(openai_tools)} 个工具就绪\n", file=sys.stderr)

    return session, openai_tools


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个天气查询助手。你可以多次调用 get_weather 工具查询不同城市的天气。\n"
    "\n"
    "工作方式（ReAct 循环）：\n"
    "1. 分析用户问题，判断需要查询哪些城市\n"
    "2. 调用 get_weather 查询城市天气\n"
    "3. 拿到结果后，判断是否需要更多数据：\n"
    "   - 如果还需要查其他城市，继续调用工具\n"
    "   - 如果信息已经足够回答用户，直接给出最终答案，不要再调用工具\n"
    "\n"
    "注意：同一轮你可以同时查询多个城市（并行调多个工具），也可以逐轮查询。"
    "请确保在信息足够时立即停止调用工具，给出清晰完整的答案。"
)

# ═══════════════════════════════════════════════════════════════════════════════
# ReAct 多轮循环（核心逻辑 —— 与 FC 版本相同，仅工具执行方式不同）
# ═══════════════════════════════════════════════════════════════════════════════

async def run(client, model: str, question: str, session: ClientSession,
              openai_tools: list[dict], max_iterations: int = 10,
              verbose: bool = True) -> dict:
    """
    MCP ReAct Agent 多轮循环：

    与 Function Call 版本循环逻辑完全一致，差异仅在工具执行层：
    - FC：直接调后端函数 get_weather(city)
    - MCP：跨进程 call_tool → session.call_tool("get_weather", {city: ...})
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    iteration = 0

    while iteration < max_iterations:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            # ── LLM 决定继续调工具 ──
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                tool_call_log.append({
                    "name": name, "args": args,
                    "iteration": iteration + 1,
                })
                if verbose:
                    print(f"  → [轮{iteration + 1}] {name}({args})")

                # MCP 方式：跨进程 call_tool
                call_result = await session.call_tool(name, args)
                result = "\n".join(
                    b.text for b in call_result.content
                    if hasattr(b, "text")
                )

                preview = (result or "")[:120].replace("\n", " ")
                if verbose:
                    print(f"    ↩ [weather] {preview}"
                          f"{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            iteration += 1
            continue  # 回到循环，LLM 可继续调工具

        else:
            # ── LLM 决定停止 ──
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 最终回答（{elapsed:.1f}s，"
                      f"共 {iteration} 轮工具调用）")
            return {
                "answer": answer,
                "tool_calls": tool_call_log,
                "iterations": iteration,
                "elapsed": elapsed,
            }

    elapsed = time.time() - t0
    if verbose:
        print(f"  ⚠ 达到最大迭代次数 {max_iterations}，强制停止")
    return {
        "answer": "（已达到最大工具调用轮数，强制停止。请尝试简化问题或增大 --max-iterations）",
        "tool_calls": tool_call_log,
        "iterations": iteration,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 问题
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    "北京、上海、广州、深圳、杭州这五个城市，今天哪个最热？",
    "比较东京和纽约今天的天气，哪个更适合户外活动？",
    "今天宁德的天气怎么样？",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

async def main_async(provider: str, question: str | None, demo: bool,
                     max_iterations: int, quiet: bool):
    client, model = build_client(provider)
    if not quiet:
        print(f"[MCP ReAct] provider={provider}  model={model}  "
              f"max_iter={max_iterations}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        session, openai_tools = await connect_server(stack)

        questions = (
            DEMO_QUESTIONS if demo
            else ([question] if question else [DEMO_QUESTIONS[0]])
        )
        for i, q in enumerate(questions, 1):
            if not quiet:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run(
                client, model, q, session, openai_tools,
                max_iterations=max_iterations,
                verbose=not quiet,
            )
            if not quiet:
                print("\n最终回答：")
                print(result["answer"])
                print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="方式二：MCP ReAct Agent（天气多轮循环）",
    )
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置 demo 问题集")
    parser.add_argument("--provider", default="deepseek",
                        choices=list(PROVIDERS.keys()))
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大工具调用轮数（默认 10，仅作安全兜底）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(
        args.provider, args.question, args.demo,
        args.max_iterations, args.quiet,
    ))


if __name__ == "__main__":
    main()
