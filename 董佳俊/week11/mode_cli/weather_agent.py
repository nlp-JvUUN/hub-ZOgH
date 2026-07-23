"""
weather_agent.py — 方式三：CLI ReAct Agent（天气查询多轮循环）

教学重点：
  1. CLI 作为"工具实现层"的 ReAct 循环：工具不是 Python 函数，而是 fincli 命令
  2. 白名单 enum 形态（run_cli）：command 限定为 weather，安全可控
  3. 与 FC/MCP 版本对比：工具定义 = argparse 子命令；执行 = subprocess 子进程

使用方式：
  # 先把 fincli 装成命令（一次即可）
  pip install -e .
  # 单问题
  python mode_cli/weather_agent.py -q "北京上海广州哪个最热？"
  # 内置 demo
  python mode_cli/weather_agent.py --demo

环境变量：
  DEEPSEEK_API_KEY（默认 LLM） / DASHSCOPE_API_KEY（备选 LLM）
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

# fincli 真实命令探测：装了就用 fincli，没装退回 python 直接跑
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

sys.path.insert(0, str(BASE_DIR))

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
# 形态 A：具名 run_cli（白名单 enum — 安全可控）
# ═══════════════════════════════════════════════════════════════════════════════

NAMED_COMMANDS = {
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}


def run_named(command: str, args: dict) -> str:
    """按白名单拼出 argv，子进程执行，返回 stdout。"""
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return (
            f"[run_cli] 未知命令：{command}"
            f"（白名单：{list(NAMED_COMMANDS)})"
        )

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"

    if proc.returncode != 0:
        return (
            f"[run_cli] 命令失败（code={proc.returncode}）："
            f"{proc.stderr[-500:]}"
        )
    return proc.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 Schema（只暴露 run_cli，command 限定 weather）
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。"
                "可用命令：weather（查询城市天气，需传 {\"city\": \"城市名\"}）"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "weather：查询指定城市的天气",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。weather: {\"city\": \"城市中文名\"}",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def execute_tool(tool_args: dict) -> str:
    """工具执行入口：command → run_named 分发。"""
    return run_named(tool_args["command"], tool_args.get("args", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个天气查询助手。通过 run_cli 工具调用预批准的命令行工具查询天气。\n"
    "\n"
    "可用命令：weather — 查询城市天气\n"
    "  用法：run_cli(command='weather', args={'city': '北京'})\n"
    "\n"
    "工作方式（ReAct 循环）：\n"
    "1. 分析用户问题，判断需要查询哪些城市\n"
    "2. 调用 run_cli 查询城市天气\n"
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
    CLI ReAct Agent 多轮循环：

    与 FC/MCP 版本循环逻辑一致，差异仅在工具执行层：
    - FC：直接调后端函数 get_weather(city)
    - MCP：跨进程 call_tool → session.call_tool(...)
    - CLI：subprocess 执行 fincli weather --city ...
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
                args = json.loads(tc.function.arguments or "{}")
                tool_call_log.append({
                    "name": tc.function.name,
                    "args": args,
                    "iteration": iteration + 1,
                })
                if verbose:
                    print(f"  → [轮{iteration + 1}] "
                          f"run_cli(command={args.get('command')}, "
                          f"args={args.get('args')})")

                try:
                    result = execute_tool(args)
                except Exception as e:
                    result = f"[cli] 执行异常：{e}"

                preview = (result or "")[:120].replace("\n", " ")
                if verbose:
                    print(f"    ↩ {FINCLI_LABEL} {preview}"
                          f"{'...' if len(result or '') > 120 else ''}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            iteration += 1
            continue  # 回到循环，LLM 可继续调工具

        else:
            # ── LLM 决定停止 ──
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
# Demo 问题
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    "北京、上海、广州、深圳、杭州这五个城市，今天哪个最热？",
    "比较东京和纽约今天的天气，哪个更适合户外活动？",
    "今天宁德的天气怎么样？",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="方式三：CLI ReAct Agent（天气多轮循环）",
    )
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置 demo 问题集")
    parser.add_argument("--provider", default="deepseek",
                        choices=list(PROVIDERS.keys()))
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大工具调用轮数（默认 10，仅作安全兜底）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.quiet:
        print(f"[CLI ReAct] provider={args.provider}  model={model}  "
              f"max_iter={args.max_iterations}\n", file=sys.stderr)

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
