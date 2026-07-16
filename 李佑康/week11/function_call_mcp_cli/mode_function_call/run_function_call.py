"""
run_function_call.py — Function Call 方式：天气查询（循环交互）

使用方式：
  python mode_function_call/run_function_call.py
  python mode_function_call/run_function_call.py --provider dashscope

交互流程：
  用户输入城市名 → LLM 调用 get_weather 工具 → 返回天气报告 → 循环
  输入 exit / quit 退出

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY（默认 LLM）或 DASHSCOPE_API_KEY（备选）
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_backend import get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

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


# ── 工具 Schema & Dispatch ────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_weather": get_weather,
}

SYSTEM_PROMPT = (
    "你是一名天气查询助手。用户会告诉你城市名，请调用 get_weather 工具查询天气，"
    "然后用自然语言总结返回的天气报告。本回合你可以一次调用多个工具。"
)


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

def run_once(client, model: str, question: str, verbose: bool = True) -> str:
    """单轮闭环：提问 → tool_call → 执行 → 回填 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  → [tool] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            if verbose:
                preview = (result or "")[:120].replace("\n", " ")
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto",
        )
        msg = resp.choices[0].message

    return msg.content or ""


# ── 循环交互入口 ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="天气查询 — Function Call 方式")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"天气查询助手（Function Call）| provider={args.provider} model={model}")
    print("输入城市名查询天气，输入 exit / quit 退出\n")

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

        answer = run_once(client, model, question)
        print(f"\n助手：{answer}\n")


if __name__ == "__main__":
    main()
