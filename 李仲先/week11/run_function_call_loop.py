"""
run_function_call_loop.py — 方式一（循环调用版）：Function Call + 多轮 ReAct 循环

与 run_function_call.py 的区别：
  单轮版：提问 → 工具调用（一轮）→ 最终回答
  循环版：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答（最多 max_rounds 轮）

教学重点：
  1. 多轮 ReAct 循环让模型能根据工具结果决定下一步（如：查北京天气→若下雨→再查上海）
  2. 循环终止条件：模型不再输出 tool_calls，或达到 max_rounds 上限
  3. 与单轮版共享同一份 TOOLS_SCHEMA / TOOL_DISPATCH，仅 run() 不同

使用方式：
  python mode_function_call/run_function_call_loop.py --demo
  python mode_function_call/run_function_call_loop.py -q "北京天气如何？如果下雨请再查上海"
  python mode_function_call/run_function_call_loop.py --loop 5 -q "对比北京和上海的天气"
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies
from src.weather_backend import get_weather

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


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
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
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_weather": get_weather,
}


# ── 多轮循环闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。你可以多次调用工具来获取信息，然后基于工具返回的结果"
    "决定是否需要调用更多工具，直到你收集到足够信息再给出最终回答。"
    "回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。你可以根据工具返回结果决定下一步操作。"
)


def run_loop(client, model: str, question: str, max_rounds: int = 5, verbose: bool = True) -> dict:
    """
    多轮 ReAct 循环：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答。
    循环终止条件：模型不再输出 tool_calls，或达到 max_rounds 上限。
    返回 {answer, tool_calls, elapsed, rounds}。
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
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型没有调用工具 → 已有最终回答，结束循环
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

        # 模型调用了工具 → 执行并回填，继续循环
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": rounds})
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
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 达到 max_rounds 仍未结束 → 强制让模型给出回答
    if verbose:
        print(f"\n  [round {max_rounds}] 达到最大轮次，强制结束循环")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call（多轮循环版）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--loop", type=int, default=5, help="最大循环轮次（默认5）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call Loop] provider={args.provider} model={model} max_rounds={args.loop}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run_loop(client, model, q, max_rounds=args.loop, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print(f"\n共调用 {len(result['tool_calls'])} 次工具，{result['rounds']} 轮循环")
            print("最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
