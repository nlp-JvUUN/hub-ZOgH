"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. **Agent Loop 多轮闭环**：宿主不做决策，只负责 执行→回传→再问，模型自己决定
     下一步调用什么工具，直到模型不再 tool_call 为止
  3. 并行工具调用：模型一次输出多个 tool_call（如同时查年报+查坐标），宿主逐个执行后一并回填
  4. 链式调用：上一轮工具的输出（如 get_city_coordinates 返回的坐标）在下一轮可见，
     模型可以基于它调用 get_weather_by_coords
  5. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德时代2023年营收和净利润？"

  # 内置示例问题（演示并行工具调用 + 链式调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM agent loop 代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 类似，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_city_coordinates, get_weather_by_coords  # noqa: E402

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
# 天气查询拆为两个工具，模型必须分步调用——这是 agent loop 链式调用的教学用例。

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
                            "把公司名写进 query 会稀释检索精度。"
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
    # ── 天气查询拆为两个工具，强制分步链式调用 ──
    {
        "type": "function",
        "function": {
            "name": "get_city_coordinates",
            "description": (
                "根据城市中文名查询经纬度坐标。"
                "返回 JSON 包含 latitude/longitude/name/country/admin1 字段。"
                "查询天气前必须先调用本工具获取坐标，再调用 get_weather_by_coords。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'、'北京'、'上海'",
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
                "根据经纬度坐标查询当前天气及未来3天预报。"
                "需要先通过 get_city_coordinates 获取坐标，再将返回的 latitude/longitude 传入。"
                "location_str 可选，建议传入 get_city_coordinates 返回的 country+admin1+name 拼接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度，来自 get_city_coordinates 返回的 latitude 字段",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度，来自 get_city_coordinates 返回的 longitude 字段",
                    },
                    "location_str": {
                        "type": "string",
                        "description": "可选，位置描述（如'中国 福建省 宁德市'），用于天气报告标题。从 get_city_coordinates 返回的 country/admin1/name 拼接",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_city_coordinates": get_city_coordinates,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── Agent Loop（多轮闭环）──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时，分两步链式调用：先调 get_city_coordinates 获取城市经纬度，"
    "再根据返回的 latitude/longitude 调用 get_weather_by_coords 获取天气。"
    "你可以在一轮中同时调用多个不相互依赖的工具（如同时查年报和城市坐标），"
    "依赖前一轮结果的工具在下一轮调用。你可以进行多轮工具调用，直到信息足够回答问题。"
)

MAX_ITERATIONS = 10  # 安全上限，防止死循环


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    Agent Loop 多轮闭环：
      提问 → 模型决定调用工具 → 宿主执行 → 结果回填 → 模型再决定
      → ... 循环直到模型不再 tool_call（给出最终回答）。

    宿主不参与决策——只忠实地执行工具、回传结果、再请求 LLM。
    模型根据历史消息（包括所有工具结果）自主决定下一步。

    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再调用工具 → 最终回答，退出循环
        if not msg.tool_calls:
            break

        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)

        # 执行本轮所有工具调用（模型可能一次调用多个不依赖的工具）
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  -> [tool] 第{iteration}轮 {name}({args})")

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    raw = fn(**args)
                    # get_city_coordinates 返回 dict → 序列化为 JSON 字符串
                    # 其他工具返回 str → 直接使用
                    if isinstance(raw, dict):
                        result = json.dumps(raw, ensure_ascii=False)
                    else:
                        result = str(raw)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    <- {preview}{'...' if len(result or '') > 120 else ''}\n")

            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 循环继续——模型在下一轮看到全部工具结果，自主决定再调什么

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  -> [llm] 最终回答（{elapsed:.1f}s，共{len(tool_call_log)}次工具调用，{iteration}轮）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",  # 幻觉控制：比亚迪不在知识库
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
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
            _safe_print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        _safe_print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def _safe_print(s: str):
    """安全打印，兼容 Windows GBK 终端无法显示 emoji/特殊 Unicode 的情况。"""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))


if __name__ == "__main__":
    main()
