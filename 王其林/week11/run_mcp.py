#!/usr/bin/env python3
"""
run_mcp.py — 方式二：MCP（模型上下文协议）客户端 + LLM 集成

教学重点：
  1. MCP 服务器作为独立的子进程运行（stdio），客户端通过 SDK 连接。
  2. 从服务器动态获取工具列表，并转换为 OpenAI 兼容的 tools schema。
  3. 多轮对话循环：LLM 决策 → 通过 MCP 调用工具 → 回填结果 → 继续，直到无工具调用。
  4. 与 Function Call 模式采用完全相同的 LLM 交互逻辑，但工具执行由 MCP 代理，实现了协议标准化。
  5. 展示如何将第三方工具封装成 MCP 服务，并集成到现有 LLM 工作流中。

使用方式：
  # 设置 API Key
  export DEEPSEEK_API_KEY=sk-xxx   # 或 DASHSCOPE_API_KEY

  # 单问题
  python run_mcp.py --question "成都未来三天天气怎么样？"

  # 运行示例问题集
  python run_mcp.py --demo

  # 指定 provider
  python run_mcp.py --provider dashscope --question "武汉的坐标是多少？"

依赖：
  pip install mcp openai httpx
  需要同目录下有 weather_server.py（或通过路径指定）
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack

from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

BASE_DIR = Path(__file__).parent.parent

# ---------- LLM 配置（与 function_call 保持一致） ----------
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

# ── Server 配置 ────────────────────────────────────────────────────────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    # 两个自写 Server，都用项目内 Python 脚本启动，stdio 通信
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


# ── 连接所有 Server：一次走完 建管道→握手→发现工具→转 schema ───────────────

async def connect_all_servers(stack: AsyncExitStack):
    """
    连接所有 MCP Server，返回 (tool_registry, openai_tools)：
      tool_registry : tool_name → (ClientSession, server_label)，用于路由 call_tool
      openai_tools  : 转成 OpenAI tools schema 的列表，直接喂给 LLM
    """
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        # stdio_client 建立进程间通信管道（子进程的 stdin/stdout）
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))

        # initialize() = MCP 握手，协商协议版本和能力
        await session.initialize()

        # list_tools() = 工具发现；同时把 MCP inputSchema 适配成 OpenAI parameters
        # —— 这一步是"协议层 → 模型层"的转换：MCP 让工具与模型解耦，
        #   但喂给具体 LLM 时仍要变成它认识的格式（inputSchema 本就是 JSON Schema，直接塞）
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
    "你是一个天气查询助手。你可以调用两个工具：\n"
    "1. get_coordinates(city): 根据城市名获取经纬度（返回 JSON）。\n"
    "2. get_weather(latitude, longitude): 根据经纬度获取天气。\n"
    "如果用户询问某个城市的天气，你应该先调用 get_coordinates，再调用 get_weather。\n"
    "只依据工具返回的数据作答，不要编造。如果工具返回错误，请告知用户。"
)


async def run(client: OpenAI, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    """单轮闭环：提问 → 模型输出 tool_call → 路由到 Server 执行 → 回填 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    start_time = time.time()
    tool_call_log = []

    # 4. 多轮循环
    while True:
        # 调用 LLM
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())  # 保留 assistant 消息

        if not msg.tool_calls:
            break  # 没有工具调用，结束

        # 5. 并发执行所有工具调用（通过 MCP）
        # 为了简单，这里使用 gather 并发
        tool_calls = msg.tool_calls
        if verbose:
            print(f"[MCP] 执行 {len(tool_calls)} 个工具调用")

        # 准备并发任务
        async def call_one(tc):
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            # 查路由表找到对应 Server 的 ClientSession，跨进程调用
            session, label = tool_registry.get(tool_name, (None, None))
            if verbose:
                print(f"  → [MCP] {tool_name}({args})")
            try:
                result = await session.call_tool(tool_name, args)
                # 提取文本内容（假设返回 TextContent）
                content = ""
                for item in result.content:
                    if item.type == "text":
                        content += item.text
                # 截断过长的结果
                if len(content) > 2000:
                    content = content[:2000] + "...(截断)"
                return tc.id, content
            except Exception as e:
                error_msg = f"MCP 调用失败：{e}"
                if verbose:
                    print(f"  ✗ {error_msg}")
                return tc.id, error_msg

        # 并发执行
        results = await asyncio.gather(*[call_one(tc) for tc in tool_calls])

        # 6. 回填工具结果
        for tc_id, content in results:
            tool_call_log.append({
                "name": next(tc.function.name for tc in tool_calls if tc.id == tc_id),
                "args": json.loads(next(tc.function.arguments for tc in tool_calls if tc.id == tc_id))
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": content,
            })

    # 结束循环，得到最终回答
    answer = msg.content or ""
    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
    }

# ---------- 工具 Schema 转换 ----------
def mcp_tool_to_openai_schema(tool):
    """将 MCP 工具定义转换为 OpenAI 的 function 格式"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,  # 已经是 JSON Schema
        }
    }
            

# ---------- 示例问题和入口 ----------
DEMO_QUESTIONS = [
    "成都未来三天天气怎么样？",
    "武汉具体的经纬度坐标是多少？",
    "武汉和成都未来三天天气对比",
    "成都和武汉地理位置关系",
]

async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run(client, model, q, tool_registry, openai_tools,
                               verbose=verbose and not as_json)
            result["question"] = q
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()
    asyncio.run(main_async(args.provider, args.question, args.demo, verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()