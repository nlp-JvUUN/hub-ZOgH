"""
run_function_call_v2.py — 方式一 v2：Function Call（Agent 循环 + 新知识库 + 拆分天气工具）

相对 run_function_call.py 的三处改动：
  1. 天气从"一个工具内部两次 HTTP 请求"拆成两个独立工具：
     geocode_city（城市名→经纬度）与 get_weather_by_coords（经纬度→天气），
  2. 知识库换成 src/rag_backend_v2.py：美的集团(000333)/比亚迪(002594)/招商银行(600036)，
     年份仍是 2021-2023。TOOLS_SCHEMA 的 description 与 SYSTEM_PROMPT 同步更新。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend_v2 import search_annual_report, list_companies  # noqa: E402
from src.weather_backend_v2 import geocode_city, get_weather_by_coords  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",  # 即 deepseek-v4-flash
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
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
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 3 家公司：美的集团(000333)/比亚迪(002594)/"
                "招商银行(600036)，年份仅 2021/2022/2023。"
                "不在库内的公司请勿调用本工具，应直接告知用户不在知识库范围内。"
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
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤：000333(美的集团)/002594(比亚迪)/600036(招商银行)。不传则跨公司检索",
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
            "name": "geocode_city",
            "description": (
                "查询城市名对应的经纬度坐标，天气查询的第一步。"
                "返回的 JSON 里含 latitude/longitude，需要作为 get_weather_by_coords 的输入参数。"
                "不要跳过这一步直接猜坐标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'深圳'"},
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
                "根据经纬度查询当前天气及未来3天预报，天气查询的第二步。"
                "latitude/longitude 必须先调用 geocode_city 获取，不要自己猜测坐标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，来自 geocode_city 的返回结果"},
                    "longitude": {"type": "number", "description": "经度，来自 geocode_city 的返回结果"},
                    "location_name": {
                        "type": "string",
                        "description": "可选，地名展示文本（如 '福建省 宁德市'），来自 geocode_city 的 country+admin1+city_name",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "geocode_city": geocode_city,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── Agent 循环 ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（美的集团/比亚迪/招商银行），请明确告知不在库内，不要臆测。"
    "涉及天气时，必须先调用 geocode_city 拿到经纬度，再调用 get_weather_by_coords 查询天气，"
    "不要跳过第一步直接猜坐标。你可以在一轮里同时发出多个工具调用，也可以分多轮逐步调用，"
    "直到收集到足够信息为止，再给出最终回答。"
)

MAX_TURNS = 6  # Agent 循环安全上限，避免模型死循环调用工具


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    Agent 循环：提问 → 模型输出 tool_call → 执行 → 回填 → 再问模型，
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    turn = 0
    msg = None
    while turn < MAX_TURNS:
        turn += 1
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # 模型不再要工具，本轮 content 就是最终回答，退出循环
            break

        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [turn {turn}][tool] {name}({args})")
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
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 循环继续：下一轮把工具结果给模型，模型可能再要工具，也可能直接给答案

    answer = (msg.content or "") if msg else ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，共 {turn} 轮）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed, "turns": turn}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "美的集团2023年营收和净利润是多少？",
    "美的集团2023年营收和净利润是多少？另外总部佛山的天气如何？",  # 演示 RAG + 两步天气工具串联
    "对比比亚迪和招商银行2023年的营收。",
    "贵州茅台2023年营收是多少？",  # 幻觉控制：贵州茅台已不在新知识库
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一 v2：Function Call（Agent 循环）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call v2 / Agent] provider={args.provider} model={model}\n")

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
