"""
compare_weather_loop.py - 天气循环调用三方式对比运行器

对同一个天气问题，依次运行：
  1. Function Call 多轮循环
  2. MCP 多轮循环
  3. CLI 多轮循环

记录工具调用顺序、调用轮数、耗时和最终答案摘要，并写入：
  output/compare_weather_loop_result.md

运行：
  python compare_weather_loop.py
  python compare_weather_loop.py --questions "查询宁德天气" "查询北京天气"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).parent
PY = sys.executable

MODES = [
    (
        "Function Call",
        [PY, str(BASE_DIR / "mode_function_call" / "run_function_call_weather_loop.py"), "--json", "--quiet"],
    ),
    (
        "MCP",
        [PY, str(BASE_DIR / "mode_mcp" / "run_mcp_weather_loop.py"), "--json", "--quiet"],
    ),
    (
        "CLI",
        [PY, str(BASE_DIR / "mode_cli" / "run_cli_weather_loop.py"), "--json", "--quiet"],
    ),
]

DEFAULT_QUESTIONS = ["查询宁德天气"]
EXPECTED_STEPS = ["get_location", "get_weather_by_location"]
WEATHER_ANSWER_HINTS = ["天气", "温度", "湿度", "风速", "预报"]


def run_one(mode_cmd: list[str], question: str, provider: str, max_rounds: int, cities: list[str] | None) -> dict:
    cmd = mode_cmd + ["--provider", provider, "--max-rounds", str(max_rounds), "-q", question]
    if cities:
        cmd += ["--cities", *cities]
    started = time.time()
    child_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            cwd=str(BASE_DIR),
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时（>180s）", "wall_elapsed": time.time() - started}
    except Exception as exc:
        return {"ok": False, "error": f"子进程启动/读取失败：{exc}", "wall_elapsed": time.time() - started}

    wall_elapsed = time.time() - started
    if proc.returncode != 0:
        error_text = proc.stderr or proc.stdout
        if len(error_text) > 2400:
            error_text = error_text[:1600] + "\n...\n" + error_text[-800:]
        return {"ok": False, "error": error_text, "wall_elapsed": wall_elapsed}

    out_lines = proc.stdout.strip().splitlines()
    if not out_lines:
        return {"ok": False, "error": "无输出", "wall_elapsed": wall_elapsed}

    try:
        data = json.loads(out_lines[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSON 解析失败：{exc}; 输出：{proc.stdout[-500:]}", "wall_elapsed": wall_elapsed}

    data["ok"] = True
    data["wall_elapsed"] = wall_elapsed
    return data


def normalize_tool_name(call: dict) -> str:
    name = call.get("name", "")
    args = call.get("args", {})
    if name == "run_cli":
        command = args.get("command")
        if command == "location":
            return "get_location"
        if command == "weather":
            return "get_weather_by_location"
    return name


def summarize(data: dict) -> dict:
    if not data.get("ok"):
        return {
            "tool_sequence": "-",
            "tool_count": 0,
            "round_count": 0,
            "completed": False,
            "llm_elapsed": "-",
            "answer_preview": "(失败) " + data.get("error", "")[:80].replace("\n", " "),
        }

    tool_calls = data.get("tool_calls", [])
    normalized = [normalize_tool_name(call) for call in tool_calls]
    answer = data.get("answer", "")
    completed = (
        all(step in normalized for step in EXPECTED_STEPS)
        and any(hint in answer for hint in WEATHER_ANSWER_HINTS)
    )
    rounds = sorted({call.get("round") for call in tool_calls if call.get("round") is not None})

    return {
        "tool_sequence": " -> ".join(normalized) or "(无工具调用)",
        "tool_count": len(tool_calls),
        "round_count": len(rounds),
        "completed": completed,
        "llm_elapsed": f"{data.get('elapsed', 0):.1f}s",
        "answer_preview": answer[:180].replace("\n", " ") + ("..." if len(answer) > 180 else ""),
        "tool_result_preview": " / ".join(
            str(item.get("result_preview", "")).replace("\n", " ").replace("|", "/")[:120]
            for item in data.get("tool_results", [])
        ),
    }


def run_compare(questions: list[str], provider: str, max_rounds: int, cities: list[str] | None) -> list[dict]:
    rows = []
    for index, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}\nQ{index}: {question}\n{'=' * 70}")
        for mode_name, mode_cmd in MODES:
            print(f"  > {mode_name} ...", end=" ", flush=True)
            data = run_one(mode_cmd, question, provider, max_rounds, cities)
            summary = summarize(data)
            status = "OK" if data.get("ok") else "FAIL"
            done = "完成" if summary["completed"] else "未完成"
            print(f"{status} {done} 工具[{summary['tool_count']}] {summary['llm_elapsed']}")
            if not data.get("ok"):
                print(f"    错误：{data.get('error', '')[:1000].replace(chr(10), ' ')}")
            rows.append({
                "question": question,
                "mode": mode_name,
                "ok": data.get("ok", False),
                **summary,
            })
    return rows


def write_markdown(rows: list[dict], questions: list[str], provider: str, cities: list[str] | None, path: Path) -> None:
    lines = [
        "# 天气循环调用三方式对比结果",
        "",
        f"- LLM provider: `{provider}`",
        f"- 问题数: {len(questions)}",
        f"- 方式数: {len(MODES)}",
        f"- 城市列表: `{', '.join(cities) if cities else '从问题文本解析'}`",
        "- 目标链路: `get_location -> get_weather_by_location -> 最终天气回答`",
        "",
        "## 对比表",
        "",
        "| 问题 | 方式 | 工具调用顺序 | 工具数 | 循环轮数 | LLM耗时 | 是否完成两步天气查询 | 答案摘要 |",
        "|------|------|--------------|:------:|:--------:|:-------:|:--------------------:|----------|",
    ]

    for row in rows:
        completed = "✓ 完成" if row["completed"] else "✗ 未完成"
        lines.append(
            f"| {row['question']} | {row['mode']} | {row['tool_sequence']} | "
            f"{row['tool_count']} | {row['round_count']} | {row['llm_elapsed']} | "
            f"{completed} | {row['answer_preview']} |"
        )

    lines += [
        "",
        "## 工具结果预览",
        "",
        "| 问题 | 方式 | 工具结果摘要 |",
        "|------|------|--------------|",
    ]
    for row in rows:
        lines.append(f"| {row['question']} | {row['mode']} | {row.get('tool_result_preview', '') or '-'} |")

    lines += [
        "",
        "## 解读",
        "",
        "- **Function Call**: 工具 schema 手写，执行时直接调用 Python 函数，路径最短，适合快速验证多轮工具决策。",
        "- **MCP**: 工具由 Server 暴露，Host 通过 `list_tools` 发现并通过 `call_tool` 调用，复用性最好，但有协议和子进程开销。",
        "- **CLI**: 底层能力先封装为命令行，Host 通过子进程执行，最接近真实工程命令工具，调试直观但有 subprocess 开销。",
        "- 三种方式业务目标相同：都需要先拿城市经纬度，再按经纬度查天气。差异主要来自工具暴露方式、调用边界和执行成本。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="天气循环调用三方式对比运行器")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--cities", nargs="+", help="多个城市，如 深圳 北京 上海；也支持 深圳,北京,上海")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "dashscope"])
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    print(f"[compare_weather_loop] provider={args.provider}, {len(args.questions)} 个问题 × {len(MODES)} 种方式")
    rows = run_compare(args.questions, args.provider, args.max_rounds, args.cities)

    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "compare_weather_loop_result.md"
    write_markdown(rows, args.questions, args.provider, args.cities, out_path)

    print(f"\n对比结果已写入：{out_path}")


if __name__ == "__main__":
    main()
