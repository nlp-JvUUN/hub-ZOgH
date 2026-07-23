"""
run_weather_function_call.py — 纯天气 Function Call

教学重点：
  1. 手写 JSON Schema：weather 工具的 name/description/parameters 开发者自己写
  2. 单轮闭环三步：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填
  3. 工具名 → 后端函数的 dispatch 表：业务逻辑与协议层彻底分离

使用方式：
  python weather_mode_function_call/run_weather_function_call.py -q "宁德的天气如何？"
  python weather_mode_function_call/run_weather_function_call.py --demo

依赖：
  pip install openai
  环境变量：DEEPSEEK_API_KEY / DASHSCOPE_API_KEY
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_mode_function_call.weather_backend import (  # noqa: E402
    get_coordinates,
    get_weather_by_coords,
)

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


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": (
                "根据城市名获取经纬度坐标。城市用中文名，如 '宁德'、'北京'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": (
                "根据经纬度查询城市的当前天气及未来3天预报。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "纬度，如 26.6592",
                    },
                    "lon": {
                        "type": "number",
                        "description": "经度，如 119.5477",
                    },
                    "city_name": {
                        "type": "string",
                        "description": "城市名称（用于显示），如 '宁德'",
                    },
                },
                "required": ["lat", "lon", "city_name"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────

TOOL_DISPATCH = {
    "get_coordinates": get_coordinates,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气助手。可以调用以下工具：\n"
    "1. get_coordinates - 根据城市名获取经纬度\n"
    "2. get_weather_by_coords - 根据经纬度查询天气\n"
    "只依据工具返回的数据作答，不要编造天气信息。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    step = 0

    while True:
        step += 1
        print(f"\n{'='*60}")
        print(f"【第 {step} 步】发送消息给 LLM")
        print(f"{'='*60}")
        print(f"messages 数量: {len(messages)}")
        for i, m in enumerate(messages):
            if isinstance(m, dict):
                role = m['role']
                content_len = len(str(m.get('content', '')))
            else:
                role = m.role
                content_len = len(str(m.content or ''))
            print(f"  [{i}] role={role}, content长度={content_len}")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        print(f"\n--- LLM 返回 (step {step}) ---")
        print(f"content: {msg.content}")
        tc_list = msg.tool_calls if msg.tool_calls else []
        print(f"tool_calls 数量: {len(tc_list)}")
        for tc in tc_list:
            print(f"  - {tc.function.name}: {tc.function.arguments}")

        if not msg.tool_calls:
            print("\n>>> LLM 没有调用工具，结束")
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"\n>>> 调用工具: {name}")
                print(f"    参数: {args}")
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
                print(f"    结果:\n{result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    answer = msg.content or ""
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"【最终回答】（耗时 {elapsed:.1f}s）")
    print(f"{'='*60}")
    print(answer)
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德的经纬度是多少？",
    "宁德的天气如何？先获取经纬度，再用经纬度查天气。",
    "北京天气怎么样？先获取经纬度，再用经纬度查天气。",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="天气 Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例")
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=PROVIDERS.keys(),
    )
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Weather Function Call] provider={args.provider} model={model}\n")

    questions = (
        DEMO_QUESTIONS if args.demo
        else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    )
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(
            results[0] if len(results) == 1 else results,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
