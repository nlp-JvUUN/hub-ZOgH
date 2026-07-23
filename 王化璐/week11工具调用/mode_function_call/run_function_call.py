"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

核心设计：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 多轮闭环：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填 → 循环直到不再调用工具 → 模型生成最终回答
  3. 并行工具调用：模型一次输出多个 tool_call（如同时查AI知识+查天气），宿主逐个执行后一并回填
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "Transformer自注意力机制原理是什么？"

  # 内置示例问题（演示并行工具调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的核心差异点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_ai_knowledge, list_papers  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402

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


# ── 工具 Schema 定义 ────────────────────────────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_ai_knowledge",
            "description": (
                "在AI技术面试知识库中检索与问题最相关的段落。"
                "知识库仅收录以下论文：Attention Is All You Need(Transformer架构)/"
                "BERT(预训练语言模型)/GPT-3(大语言模型)/InstructGPT ChatGPT(指令微调)/"
                "LLaMA(开源大模型)/RAG(检索增强生成)/动手学深度学习(深度学习教程)。"
                "不在库内的主题请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含论文标题和主题"
                            "（已由 title/topic 参数过滤），只用简短技术术语，"
                            "例如 '自注意力机制'、'预训练目标'、'少样本学习'。"
                            "把标题写进 query 会稀释检索精度。"
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "可选，按论文标题过滤，如 'Attention Is All You Need'。不传则跨论文检索",
                    },
                    "topic": {
                        "type": "string",
                        "description": "可选，按主题过滤：'Transformer架构' / '预训练语言模型' / '大语言模型' / '指令微调' / '开源大模型' / '检索增强生成' / '深度学习教程'",
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
            "name": "list_papers",
            "description": "列出AI技术面试知识库中收录的所有论文、主题与年份。用于确认目标论文在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '北京'、'上海'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '北京'"},
                },
                "required": ["city"],
            },
        },
    },
]

# ── 工具路由表 ──────────────────────────────────────────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "search_ai_knowledge": search_ai_knowledge,
    "list_papers": list_papers,
    "get_weather": get_weather,
}


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名AI技术面试助手。回答用户关于AI技术的问题时，必须先调用 search_ai_knowledge 工具检索论文原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的主题不在知识库"
    "（Transformer架构/预训练语言模型/大语言模型/指令微调/开源大模型/检索增强生成/深度学习教程），"
    "请明确告知不在库内，不要臆测。涉及天气时调用 get_weather。"
    "检索完成后，必须基于返回的段落内容给出完整、详细的回答，禁止回复'让我再检索'、'再查一下'之类的话。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 循环直到不再调用工具 → 最终回答。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
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
                        result = fn(**args)
                    except TypeError as e:
                        result = f"参数错误：{e}"
                    except Exception as e:
                        result = f"工具执行失败：{e}"
                preview = (result or "")[:120].replace("\n", " ")
                if verbose:
                    print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            break

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Transformer的自注意力机制原理是什么？",
    "Transformer的自注意力机制原理是什么？另外北京的天气如何？",
    "对比BERT和GPT-3的预训练方式有什么不同？",
    "区块链的共识算法是什么？",  # 幻觉控制：区块链不在知识库
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
            print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
