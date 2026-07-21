"""
weather_agent.py — 方式一：Function Call ReAct Agent（天气查询多轮循环）

教学重点：
  1. ReAct 循环：LLM 每一轮自主决定"继续调工具"还是"给最终答案"
  2. 与单轮闭环的本质区别：while 循环替代了"调一次工具 → 回填 → 最终回答"的两步模式
  3. 只保留天气查询（get_weather），去掉年报 RAG，聚焦循环调用本身

使用方式：
  # 单问题
  python mode_function_call/weather_agent.py -q "北京上海广州哪个最热？"
  # 内置 demo（多城市比较，自然触发多轮）
  python mode_function_call/weather_agent.py --demo
  # 换成 DashScope
  python mode_function_call/weather_agent.py --demo --provider dashscope

环境变量：
  DEEPSEEK_API_KEY（默认 LLM） / DASHSCOPE_API_KEY（备选 LLM）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_backend import get_weather  # noqa: E402

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
# 工具 Schema（只保留 get_weather）
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前天气及未来3天预报。"
                "城市用中文名，如 '北京'、'宁德'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '北京'、'宁德'",
                    },
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {"get_weather": get_weather}

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt（引导 LLM 进入 ReAct 多轮思考模式）
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
# ReAct 多轮循环（核心逻辑）
# ═══════════════════════════════════════════════════════════════════════════════

def run(client, model: str, question: str, max_iterations: int = 10,
        verbose: bool = True) -> dict:
    """
    ReAct Agent 多轮循环：

    ┌─────────────────────────────────────────┐
    │  while iteration < max_iterations:      │
    │    LLM 输出 tool_calls 或最终回答        │
    │    ├── 有 tool_calls → 执行 → 回填      │
    │    │   → continue（继续循环，LLM 可再调） │
    │    └── 无 tool_calls → 最终回答 → break  │
    └─────────────────────────────────────────┘

    与单轮闭环的关键区别：
    - 单轮：第一次调工具 → 第二次直接要最终回答（固定两轮）
    - ReAct：每轮后都回到 while 开头，LLM 自主判断是否继续
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
            tools=TOOLS_SCHEMA,
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
                    print(f"    ↩ {preview}"
                          f"{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            iteration += 1
            # ── continue：回到 while 开头，LLM 可以继续调工具 ──
            continue

        else:
            # ── LLM 决定停止，给出最终答案 ──
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

    # 达到 max_iterations 上限（安全兜底）
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
# Demo 问题（设计为自然触发多轮工具调用）
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    # Q1：5 个城市比温度 — 典型多轮场景
    "北京、上海、广州、深圳、杭州这五个城市，今天哪个最热？",
    # Q2：跨国对比 — 至少 2 次调用
    "比较东京和纽约今天的天气，哪个更适合户外活动？",
    # Q3：单城市 — 验证单轮也能正常收敛
    "今天宁德的天气怎么样？",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="方式一：Function Call ReAct Agent（天气多轮循环）",
    )
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置 demo 问题集")
    parser.add_argument("--provider", default="deepseek",
                        choices=list(PROVIDERS.keys()))
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大工具调用轮数（默认 10，仅作安全兜底）")
    parser.add_argument("--quiet", action="store_true",
                        help="少输出")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.quiet:
        print(f"[Function Call ReAct] provider={args.provider}  model={model}  "
              f"max_iter={args.max_iterations}\n")

    questions = (
        DEMO_QUESTIONS if args.demo
        else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    )
    for i, q in enumerate(questions, 1):
        if not args.quiet:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q,
                     max_iterations=args.max_iterations,
                     verbose=not args.quiet)
        if not args.quiet:
            print("\n最终回答：")
            print(result["answer"])
            print()


if __name__ == "__main__":
    main()
