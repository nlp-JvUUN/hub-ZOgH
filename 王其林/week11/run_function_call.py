#!/usr/bin/env python3
"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema（现已改进为自动生成） -> 降低维护成本
  2. 多轮工具调用循环，支持连续多次调用（如先查坐标再查天气）
  3. 并发执行多个工具调用（模型一次返回多个 tool_call 时并行处理）
  4. 结果截断与错误容错，保证上下文安全
  5. 工具调用日志和结构化输出，便于与其他模式对比

使用方式：
  # 设置环境变量
  export DEEPSEEK_API_KEY=sk-xxx   # 或 DASHSCOPE_API_KEY

  # 单个问题
  python run_function_call.py --question "成都未来三天天气怎么样？"

  # 运行示例问题集
  python run_function_call.py --demo

  # 指定 provider（deepseek / dashscope）
  python run_function_call.py --provider dashscope --question "武汉坐标"

  # JSON 输出（供对比工具解析）
  python run_function_call.py --question "天气" --json

依赖：
  pip install openai httpx pydantic tenacity
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
from openai import OpenAI
from pydantic import create_model

# 将项目根目录加入 sys.path（确保能找到 src 模块）
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tool_backend import get_coordinates, get_weather  # noqa: E402

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("function_call")

# ---------- LLM Provider 配置 ----------
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

# ---------- 系统提示 ----------
SYSTEM_PROMPT = (
    "你是一个天气查询助手。你可以调用以下工具：\n"
    "1. get_coordinates(city): 根据城市名获取经纬度（返回 JSON 字符串）。\n"
    "2. get_weather(latitude, longitude): 根据经纬度获取天气信息。\n"
    "如果用户询问某个城市的天气，请先调用 get_coordinates 获取经纬度，再调用 get_weather。\n"
    "如果用户只需要坐标，只调用 get_coordinates。\n"
    "只依据工具返回的数据作答，不要编造数据。如果工具返回错误，请告知用户并建议检查输入。"
)

# ---------- 自动生成工具 Schema（基于 Pydantic） ----------
def function_to_tool(func):
    """
    根据函数签名自动生成 OpenAI 兼容的 function schema。
    支持类型提示和默认值。
    """
    import inspect
    sig = inspect.signature(func)
    fields = {}
    for name, param in sig.parameters.items():
        # 获取类型注解，默认为 str
        anno = param.annotation if param.annotation != inspect.Parameter.empty else str
        # 默认值，如果没有则为 ...
        default = ... if param.default == inspect.Parameter.empty else param.default
        fields[name] = (anno, default)
    # 动态创建 Pydantic 模型
    Model = create_model(f"{func.__name__}_args", **fields)
    schema = Model.model_json_schema()
    # 移除 $defs 等冗余字段
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        }
    }

# 生成工具列表
TOOLS_SCHEMA = [function_to_tool(get_coordinates), function_to_tool(get_weather)]
TOOL_DISPATCH = {
    "get_coordinates": get_coordinates,
    "get_weather": get_weather,
}

# ---------- 工具执行函数（支持并发） ----------
def execute_tool(tc) -> tuple[str, str]:
    """
    执行单个工具调用，返回 (tool_call_id, 结果字符串)。
    自动处理异常和结果截断。
    """
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")
    logger.info(f"执行工具 {name}({args})")
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return tc.id, f"未知工具：{name}"
    try:
        result = fn(**args)
        # 确保结果是字符串
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        # 截断过长结果（防止溢出上下文）
        if len(result) > 2000:
            result = result[:2000] + "...(截断)"
        return tc.id, result
    except Exception as e:
        logger.error(f"工具 {name} 执行失败: {e}", exc_info=True)
        return tc.id, f"工具执行失败：{e}"

# ---------- 核心问答函数 ----------
def run(client: OpenAI, model: str, question: str, verbose: bool = True) -> Dict[str, Any]:
    """
    执行单轮问答（可能包含多次工具调用循环）。
    返回：{
        "answer": str,
        "tool_calls": List[{"name": str, "args": dict}],
        "elapsed": float
    }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    start_time = time.time()
    tool_call_log = []

    while True:
        # 请求 LLM
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        # 保存 assistant 消息（含可能的 tool_calls）
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            # 无工具调用，得到最终回答
            break

        # ---------- 并发执行工具 ----------
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_tc = {
                executor.submit(execute_tool, tc): tc
                for tc in msg.tool_calls
            }
            for future in as_completed(future_to_tc):
                tc = future_to_tc[future]
                try:
                    tc_id, result = future.result()
                    # 记录日志
                    tool_call_log.append({
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments or "{}")
                    })
                    if verbose:
                        preview = (result or "")[:120].replace("\n", " ")
                        logger.info(f"  ↩ {preview}{'...' if len(result) > 120 else ''}")
                    # 回填工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    })
                except Exception as e:
                    logger.error(f"并发执行异常: {e}", exc_info=True)
                    # 仍回填错误信息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"执行失败：{e}",
                    })

        # 循环继续，LLM 会看到工具结果并决定下一步

    answer = msg.content or ""
    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
    }

# ---------- 命令行入口 ----------
DEMO_QUESTIONS = [
    "成都未来三天天气怎么样？",
    "武汉具体的经纬度坐标是多少？",
    "武汉和成都未来三天天气对比",
    "成都和武汉地理位置关系",
]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="运行内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys(),
                        help="LLM 提供商")
    parser.add_argument("--quiet", action="store_true", help="减少输出（用于对比）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式（供比较器解析）")
    args = parser.parse_args()

    # 初始化 LLM 客户端
    cfg = PROVIDERS[args.provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {args.provider.upper()}_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    if not args.json:
        print(f"[Function Call] provider={args.provider}, model={model}")

    # 准备问题列表
    if args.question:
        questions = [args.question]
    elif args.demo:
        questions = DEMO_QUESTIONS
    else:
        questions = [DEMO_QUESTIONS[0]]  # 默认

    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        try:
            result = run(client, model, q, verbose=not (args.quiet or args.json))
            result["question"] = q
            results.append(result)
            if not args.json:
                print("\n最终回答：")
                print(result["answer"])
                print(f"耗时: {result['elapsed']:.2f}s")
                if result["tool_calls"]:
                    print("工具调用记录：")
                    for tc in result["tool_calls"]:
                        print(f"  {tc['name']}({tc['args']})")
        except Exception as e:
            print(f"错误：{e}", file=sys.stderr)
            if args.json:
                print(json.dumps({"error": str(e), "question": q}, ensure_ascii=False))
            sys.exit(1)

    if args.json:
        # 输出 JSON（单问题输出对象，多问题输出数组）
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()