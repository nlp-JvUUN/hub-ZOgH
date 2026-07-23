"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 单轮循环：模型输出 tool_call → 宿主执行工具 → 模型持续判断 → 模型生成最终回答
  3. 前一个工具的输出可以作为后一个工具的输入，形成链式调用（本示例中天气查询就是两步：城市名→经纬度→天气）
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "北京今天的天气如何？"

  # 内置示例问题（演示并行工具调用）
  python run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent))

from weather_backend import get_geocoding, get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
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
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_geocoding",
            "description": "查询指定城市的经纬度信息。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定经纬度的当前天气及未来3天预报。如 '宁德' 的经纬度为 {'latitude': 24.5432, 'longitude': 119.2345'}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "object", "description": "城市经纬度信息字典，包含 latitude、longitude、name、country、admin1 等字段。"},
                },
                "required": ["location"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "get_geocoding": get_geocoding,
    "get_weather": get_weather,
}


# ── 单次循环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名地理信息查询助手。回答用户关于城市地理位置或天气相关的问题"
    "只依据工具返回的段落作答，不要编造数据。本次循环你可以多次调用工具，直到你确定用户的问题可以回答为止。"
)


def _serialize_tool_result(result) -> str:
    if result is None:
        return ""
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    单次循环：提问 → 模型输出 tool_call → 执行 → 模型判断下一个tool_call → 执行 → ... → 模型判断结束 → 输出最终回答
    1. messages 里 role=assistant 的消息可能带 tool_calls，表示模型决定调用工具。
    2. messages 里 role=tool 的消息是宿主执行工具的结果，模型看到后会生成最终回答。

    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # 第一次请求：模型看到用户问题，决定调用工具（可能多次）
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    messages.append(msg)

    # 循环执行工具调用，直到模型判断结束
    while True:
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        # 先执行模型输出的 tool_call
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 工具执行！！
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            content_text = _serialize_tool_result(result)
            preview = content_text[:120].replace("\n", " ")  # 预览结果，避免打印过长
            if verbose:
                print(f"    ↩ {preview}{'...' if len(content_text) > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content_text,
            })

        # 第二次请求：模型看到工具结果，判断是否需要调用其他工具
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "上海的地理位置是？",
    "长沙的天气如何？长沙属于哪个省份？",
    "越南的首都河内的经纬度是多少？",
    "广州的海拔是多少？",
    "北京明天会不会下雨？",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="dashscope", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
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
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
