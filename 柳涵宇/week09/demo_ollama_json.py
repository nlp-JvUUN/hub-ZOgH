"""
用 Ollama 部署 Qwen2 0.5B，并演示 JSON / JSON Schema 输出。

准备：
  1. 安装 Ollama: https://ollama.com/download
  2. 拉取模型: ollama pull qwen2:0.5b
  3. 确认服务: curl http://localhost:11434/api/tags

运行：
  python demo_ollama_json.py

说明：
  Ollama 不支持 vLLM 的 extra_body={"guided_json": ...}。
  它使用原生 /api/chat 的 format 字段：
    - format="json"：要求输出合法 JSON
    - format=<JSON Schema>：要求输出符合 schema
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
from jsonschema import ValidationError, validate


OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2:0.5b"


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer", "minimum": 2015, "maximum": 2025},
        "metric": {
            "type": "string",
            "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"],
        },
    },
    "required": ["company", "year", "metric"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是财报问答助手。从用户问题中提取结构化信息。
只输出 JSON，不要输出解释文字。

字段定义：
- company: 公司全称
- year: 年度，2015 到 2025 的整数
- metric: 必须是 ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"] 之一

示例：
{"company": "招商银行", "year": 2023, "metric": "营收"}"""


TEST_CASES = [
    "招行 2023 年营收多少",
    "贵州茅台 2022 的净利润",
    "平安银行去年（2024）的 ROE",
    "2021 年五粮液毛利率",
    "2023 宁德时代经营现金流",
    "问一下比亚迪 2024 的总资产规模",
    "茅台 2020 年利润情况",
    "ICBC 2023 营收",
    "隆基绿能 22 年 roe",
]


def check_ollama() -> None:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            "无法连接 Ollama 服务。\n"
            "请先安装并启动 Ollama，然后运行：ollama pull qwen2:0.5b\n"
            f"原始错误：{exc}"
        ) from exc

    models = [item.get("name", "") for item in resp.json().get("models", [])]
    if MODEL not in models:
        print(f"提示：当前 Ollama 模型列表里没有 {MODEL!r}。")
        print(f"请先运行：ollama pull {MODEL}")
        print()


def chat(user_msg: str, mode: str) -> tuple[str, float]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    if mode == "json":
        payload["format"] = "json"
    elif mode == "schema":
        payload["format"] = INTENT_SCHEMA

    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    dt = time.time() - t0
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip(), dt


def evaluate(output: str) -> dict[str, bool]:
    result = {
        "is_json": False,
        "has_all_fields": False,
        "year_in_range": False,
        "metric_in_enum": False,
        "schema_valid": False,
    }
    try:
        obj = json.loads(output)
        result["is_json"] = True
    except json.JSONDecodeError:
        return result

    result["has_all_fields"] = all(k in obj for k in INTENT_SCHEMA["required"])
    year = obj.get("year")
    result["year_in_range"] = isinstance(year, int) and 2015 <= year <= 2025
    result["metric_in_enum"] = obj.get("metric") in INTENT_SCHEMA["properties"]["metric"]["enum"]

    try:
        validate(instance=obj, schema=INTENT_SCHEMA)
        result["schema_valid"] = True
    except ValidationError:
        pass
    return result


def main() -> None:
    check_ollama()

    print("=" * 78)
    print("  Demo: Ollama JSON / JSON Schema 输出")
    print(f"  Ollama: {OLLAMA_HOST}")
    print(f"  Model:  {MODEL}")
    print("=" * 78)

    modes = ["raw", "json", "schema"]
    counters = {m: {"json": 0, "fields": 0, "year": 0, "enum": 0, "valid": 0} for m in modes}
    latency = {m: 0.0 for m in modes}

    for user in TEST_CASES:
        print(f"\n> {user}")
        for mode in modes:
            out, dt = chat(user, mode)
            latency[mode] += dt
            ev = evaluate(out)
            if ev["is_json"]:
                counters[mode]["json"] += 1
            if ev["has_all_fields"]:
                counters[mode]["fields"] += 1
            if ev["year_in_range"]:
                counters[mode]["year"] += 1
            if ev["metric_in_enum"]:
                counters[mode]["enum"] += 1
            if ev["schema_valid"]:
                counters[mode]["valid"] += 1

            mark = "OK" if ev["schema_valid"] else "--"
            short = out[:90] + "..." if len(out) > 90 else out
            print(f"  [{mode:<6}] {mark} {dt:6.2f}s  {short}")

    n = len(TEST_CASES)
    print("\n" + "=" * 78)
    print(f"  {n} 条测试结果汇总")
    print("=" * 78)
    print(f"{'指标':<24}{'raw':<16}{'json':<16}{'schema':<16}")
    print("-" * 78)
    rows = [
        ("合法 JSON", "json"),
        ("字段齐全", "fields"),
        ("year 在范围内", "year"),
        ("metric 在枚举内", "enum"),
        ("schema 完全通过", "valid"),
    ]
    for label, key in rows:
        line = f"{label:<22}"
        for mode in modes:
            value = counters[mode][key]
            line += f"{value}/{n} ({100 * value / n:.0f}%)   "
        print(line)

    print("\n平均延迟：")
    for mode in modes:
        print(f"  {mode:<6}: {latency[mode] / n:.2f}s")

    print("\n说明：Ollama 的 schema 输出适合本地部署体验；vLLM 的 guided_json 更适合课程里的严格约束解码对比。")


if __name__ == "__main__":
    main()
