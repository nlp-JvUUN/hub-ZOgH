"""
run_cli_loop.py — 方式三（循环调用版）：CLI + 多轮 ReAct 循环

与 run_cli.py 的区别：
  单轮版：提问 → 工具调用（一轮）→ 最终回答
  循环版：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答（最多 max_rounds 轮）

教学重点：
  1. 多轮循环 + CLI 工具：每轮仍通过 subprocess 执行 fincli 命令
  2. 形态 A（named）和形态 B（bash）均支持循环模式
  3. 沙箱检查在每轮每次工具调用时仍然生效

使用方式：
  python mode_cli/run_cli_loop.py --mode named --demo
  python mode_cli/run_cli_loop.py --mode bash -q "北京天气如何？如果下雨请再查上海"
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

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


# ── 形态 A：具名 run_cli ───────────────────────────────────────────────────

NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
    },
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}


def run_named(command: str, args: dict) -> str:
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"
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
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


# ── 形态 B：通用 run_bash（沙箱）──────────────────────────────────────────

DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bdeltree\b",
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"[>;]\s*(?:rm|del|format)\b",
    r"\bcurl\b.*\|\s*sh", r"\bwget\b.*\|\s*sh",
    r"\bsudo\b", r"\bchmod\b.*-R", r"\bchown\b.*-R",
    r"\bnc\b", r"\bnetcat\b",
    r"/etc/passwd", r"/etc/shadow",
    r"\bTaskkill\b", r"\bStop-Process\b",
]

ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}


def sandbox_check(command: str) -> str | None:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


# ── 两种形态的 tools schema ─────────────────────────────────────────────

NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。你可以多次调用本工具，根据前一次的结果"
                "决定下一次调用什么。command 只能取白名单内的值。"
                "可查 A 股年报（rag_search/list_companies）和天气（weather）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "rag_search（查年报）/ rag_list_companies（列公司）/ weather（查天气）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。rag_search: {query, stock_code?, year?, top_k?}; weather: {city}",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "你可以多次调用本工具，根据前一次的结果决定下一次执行什么。"
                "可用工具 fincli：fincli search --query '营收' --stock-code 300750 --year 2023；"
                "fincli weather --city 宁德。"
                "危险命令（rm/del/format/sudo 等）会被拦截。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]

MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}


# ── 多轮循环闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手。你可以通过 run_cli 工具多次调用预批准命令来收集信息，"
    "根据每次返回的结果决定下一步操作，直到收集到足够信息再给出最终回答。"
    "回答年报问题前必须先调用 rag_search 检索原文，只依据返回段落作答，不要编造。"
    "知识库仅含：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)。"
    "rag_search 的 query 不要含公司名/年份，用简短术语如 '营收和净利润'。"
    "不在库内的公司请明确告知。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手。你可以通过 run_bash 工具多次执行 fincli 命令来收集信息，"
    "根据每次返回的结果决定下一步操作，直到收集到足够信息再给出最终回答。"
    "查年报：fincli search --query '营收和净利润' --stock-code 300750 --year 2023。"
    "查天气：fincli weather --city 南京。"
    "回答必须依据命令返回的原文，不要编造。知识库仅含 5 家公司。"
    "不在库内的明确告知。"
)


def run_loop(client, model: str, question: str, mode: str,
             max_rounds: int = 5, verbose: bool = True) -> dict:
    """
    多轮 ReAct 循环：提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答。
    支持 named 和 bash 两种形态。
    """
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
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
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

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

        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args, "round": rounds})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")
            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })

    if verbose:
        print(f"\n  [round {max_rounds}] 达到最大轮次，强制结束循环")
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools_schema, tool_choice="auto",
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
    parser = argparse.ArgumentParser(description="方式三：CLI（多轮循环版）")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--loop", type=int, default=5, help="最大循环轮次（默认5）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI Loop/{args.mode}] provider={args.provider} model={model} max_rounds={args.loop}\n", file=sys.stderr)

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run_loop(client, model, q, args.mode, max_rounds=args.loop, verbose=not (args.quiet or args.json))
        result["question"] = q
        result["mode"] = args.mode
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
