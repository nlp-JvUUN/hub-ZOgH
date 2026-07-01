"""
vLLM Demo 运行器 — Ollama 适配版（py312 环境）

本脚本将所有 demo_*.py 的核心逻辑适配到 Ollama（qwen2.5:0.5b）运行，
通过 OpenAI 兼容 API（http://localhost:11434/v1）调用，
并生成完整运行日志保存到 outputs/run_log.md。

使用方式：
  python demo_runner.py

前提条件：
  1. Ollama 服务运行中：ollama serve
  2. 模型已下载：ollama pull qwen2.5:0.5b
  3. 依赖已安装：openai, jsonschema（conda install -n py312 openai jsonschema）
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.md")
MODEL = "qwen2.5:0.5b"
BASE_URL = "http://localhost:11434/v1"

# ── 低层 HTTP 调用（绕过 openai SDK 的 502 问题）─────────────────────────────
def chat_complete(messages, max_tokens=100, temperature=0,
                  response_format=None, extra_body=None):
    """直接发 HTTP 请求到 Ollama OpenAI 兼容 API"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def section(title):
    sep = "=" * 72
    log("")
    log(sep)
    log(f"  {title}")
    log(sep)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n")


# ── 初始化日志文件 ───────────────────────────────────────────────────────────
def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""# vLLM Demo 运行日志

> **生成时间**: {ts}
> **运行平台**: Ollama（CPU）
> **模型**: {MODEL}
> **API 地址**: {BASE_URL}

## 平台说明

本 demo 在 Windows 环境下运行，存在以下硬件限制：

| 组件 | 检测结果 | 影响 |
|------|---------|------|
| GPU | AMD Radeon Graphics（非 NVIDIA） | vLLM 需要 NVIDIA CUDA，本环境无法运行 |
| Ollama | 已安装 v0.24.0 | CPU 推理，速度较慢 |
| vLLM | 已安装 0.6.5（CPU） | guided_choice/regex/json 等 CUDA 扩展不可用 |

**支持的功能**: 裸 prompt、response_format（OpenAI 标准）
**不支持的功能**: guided_choice、guided_regex、guided_json（vLLM 私有扩展）

在原生 vLLM + NVIDIA GPU 环境下，以上三个约束解码功能可 100% 保证输出合法。

---
"""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(header)


# ─────────────────────────────────────────────────────────
# Demo 1: guided_choice — 枚举约束
# ─────────────────────────────────────────────────────────
INTENT_CHOICES = ["查股价", "查财报", "查新闻", "对比分析", "其他"]
INTENT_SYSTEM = f"""你是金融问答助手的意图路由器。
根据用户问题，判断意图属于以下哪一类，只输出类别名称，不要任何其他文字。

可选类别：{" / ".join(INTENT_CHOICES)}"""

INTENT_CASES = [
    ("查一下茅台今天多少钱", "查股价"),
    ("贵州茅台 2024 年营收多少亿", "查财报"),
    ("最近宁德时代有什么新闻", "查新闻"),
    ("对比一下招行和平安的净利润", "对比分析"),
    ("今天天气怎么样", "其他"),
    ("帮我看看 600000 的收盘价", "查股价"),
    ("招商银行去年的 ROE 是多少", "查财报"),
    ("宁德时代被限产了吗", "查新闻"),
    ("比亚迪和特斯拉哪个更强", "对比分析"),
    ("帮我订一张机票", "其他"),
    ("五粮液现在股价", "查股价"),
    ("平安保险的净利润增长率", "查财报"),
]


def demo_guided_choice():
    section("Demo 1: guided_choice — 枚举约束解码")
    log(f"Model: {MODEL}")
    log(f"Choices: {INTENT_CHOICES}")
    log("")
    log("注: guided_choice 是 vLLM 私有扩展，Ollama 不支持。此处展示裸 prompt 效果。")
    log("    vLLM 中 guided_choice 通过 FSM 约束解码可 100% 保证输出在枚举内。")

    raw_in_choices = 0
    raw_correct = 0
    results = []

    for user_msg, expected in INTENT_CASES:
        t0 = time.time()
        out = chat_complete(
            [{"role": "system", "content": INTENT_SYSTEM},
             {"role": "user", "content": user_msg}],
            max_tokens=10,
        )
        elapsed = time.time() - t0
        raw_in = out in INTENT_CHOICES
        if raw_in:
            raw_in_choices += 1
        if out == expected:
            raw_correct += 1

        flag = "✓" if out == expected else ("~" if raw_in else "✗")
        log(f"  {flag} [{expected}] {user_msg}")
        log(f"           → {out}  ({elapsed:.2f}s)")
        results.append({"user": user_msg, "expected": expected,
                         "output": out, "correct": out == expected,
                         "in_choices": raw_in, "time": elapsed})

    n = len(INTENT_CASES)
    log("")
    log(f"输出合法（在枚举内）: {raw_in_choices}/{n} ({100*raw_in_choices/n:.0f}%)")
    log(f"预测正确:            : {raw_correct}/{n} ({100*raw_correct/n:.0f}%)")
    log("")
    log("结论: guided_choice 在 vLLM 中通过约束解码 100% 保证输出合法。")
    log("      小模型 + 枚举约束 比 裸 prompt 的准确率通常更高（不会被错误 token 带偏）。")
    return results


# ─────────────────────────────────────────────────────────
# Demo 2: guided_regex — 正则约束
# ─────────────────────────────────────────────────────────
DATE_REGEX = r"\d{4}-\d{2}-\d{2}"
DATE_SYSTEM = "你是日期抽取助手。从用户输入中抽取日期，严格用 YYYY-MM-DD 格式输出，不输出任何其他文字。"
DATE_CASES = [
    "2024年5月12日",
    "2023/12/1 下午开会",
    "三月三号我去北京",
    "2024.11.30 是截止日期",
    "明天（假设今天是2026-05-11）",
    "2024 年 10 月的第一天",
]

STOCK_REGEX = r"\d{6}"
STOCK_SYSTEM = "你是股票代码抽取助手。从用户输入中找到 A 股代码（6 位数字），直接输出代码，不输出任何其他文字。"
STOCK_CASES = [
    "帮我查 600000 浦发银行",
    "code: 000001 平安银行",
    "茅台的代码是 600519",
    "六零零五一九",
    "股票代码：300750（宁德时代）",
]


def demo_guided_regex():
    section("Demo 2: guided_regex — 正则约束解码")
    log(f"Model: {MODEL}")
    log("")
    log("注: guided_regex 是 vLLM 私有扩展，Ollama 不支持。此处展示裸 prompt + 正则验证。")
    log("    vLLM 中 guided_regex 通过 FSM 约束解码可 100% 保证输出格式合法。")

    all_results = {}

    def run_section(title, system, regex, cases):
        log("")
        log(f"  — {title} —")
        log(f"  正则: {regex}")
        raw_ok = 0
        results = []

        for user in cases:
            t0 = time.time()
            out = chat_complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=30,
            )
            elapsed = time.time() - t0
            raw_match = bool(re.fullmatch(regex, out))
            if raw_match:
                raw_ok += 1

            flag = "✓" if raw_match else "✗"
            log(f"  {flag} {user}")
            log(f"      → {out}  ({elapsed:.2f}s)")
            results.append({"user": user, "output": out,
                            "match": raw_match, "time": elapsed})

        n = len(cases)
        log(f"  格式合法率: {raw_ok}/{n} ({100*raw_ok/n:.0f}%)")
        return results

    all_results["date"]  = run_section(
        "任务 1：日期标准化 → YYYY-MM-DD", DATE_SYSTEM, DATE_REGEX, DATE_CASES)
    all_results["stock"] = run_section(
        "任务 2：A 股代码抽取 → 6 位数字", STOCK_SYSTEM, STOCK_REGEX, STOCK_CASES)

    log("")
    log("结论: guided_regex 在 vLLM 中通过 FSM 约束解码 100% 保证输出格式合法。")
    log("      特别适合日期/电话/代码/邮编等有严格格式的字段。")
    return all_results


# ─────────────────────────────────────────────────────────
# Demo 3: guided_json — JSON Schema 约束
# ─────────────────────────────────────────────────────────
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string",
                    "description": "公司全称，如 招商银行、贵州茅台"},
        "year":    {"type": "integer", "minimum": 2015, "maximum": 2025},
        "metric":  {"type": "string",
                    "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"]},
    },
    "required": ["company", "year", "metric"],
    "additionalProperties": False,
}

JSON_SYSTEM = """你是财报问答助手。从用户问题中提取结构化信息，输出纯 JSON，不要任何解释文字。

字段定义：
  company: 公司全称
  year: 年度（2015~2025 整数）
  metric: 指标，必须是 ['营收', '净利润', 'ROE', '毛利率', '总资产', '经营现金流'] 之一

示例输出：
{"company": "招商银行", "year": 2023, "metric": "营收"}"""

JSON_CASES = [
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


def eval_json(output):
    result = {
        "is_json": False, "has_all_fields": False,
        "year_in_range": False, "metric_in_enum": False,
        "schema_valid": False, "parsed": None,
    }
    try:
        obj = json.loads(output)
        result["is_json"] = True
        result["parsed"] = obj
    except json.JSONDecodeError:
        return result

    required = INTENT_SCHEMA["required"]
    if all(k in obj for k in required):
        result["has_all_fields"] = True

    yr = obj.get("year")
    if isinstance(yr, int) and 2015 <= yr <= 2025:
        result["year_in_range"] = True

    if obj.get("metric") in INTENT_SCHEMA["properties"]["metric"]["enum"]:
        result["metric_in_enum"] = True

    try:
        from jsonschema import validate, ValidationError
        validate(instance=obj, schema=INTENT_SCHEMA)
        result["schema_valid"] = True
    except Exception:
        pass

    return result


def demo_guided_json():
    section("Demo 3: guided_json — JSON Schema 约束解码")
    log(f"Model: {MODEL}")
    log("")
    log("注: guided_json 是 vLLM 私有扩展，Ollama 不支持。")
    log("    vLLM 中 guided_json 通过 FSM 约束解码可 100% 保证 schema 合法。")
    log("    此处用 response_format（OpenAI 标准）替代演示。")

    raw_stats = {"is_json": 0, "has_all_fields": 0, "year_in_range": 0,
                 "metric_in_enum": 0, "schema_valid": 0}
    rf_stats  = {"is_json": 0, "has_all_fields": 0, "year_in_range": 0,
                 "metric_in_enum": 0, "schema_valid": 0}
    all_results = []

    for user in JSON_CASES:
        log("")
        log(f"  ▶ {user}")

        msgs = [{"role": "system", "content": JSON_SYSTEM},
                {"role": "user", "content": user}]

        # 裸 prompt
        t0 = time.time()
        raw_out = chat_complete(msgs, max_tokens=120)
        raw_elapsed = time.time() - t0
        raw_ev = eval_json(raw_out)
        for k in raw_stats:
            raw_stats[k] += int(raw_ev[k])
        tag = "✓" if raw_ev["schema_valid"] else "✗"
        log(f"    [裸 prompt     ] {tag} {raw_out[:80]}")
        log(f"                      ({raw_elapsed:.2f}s)")

        # response_format (Ollama 支持)
        t0 = time.time()
        rf_out = chat_complete(msgs, max_tokens=120,
                               response_format={"type": "json_object"})
        rf_elapsed = time.time() - t0
        rf_ev = eval_json(rf_out)
        for k in rf_stats:
            rf_stats[k] += int(rf_ev[k])
        tag = "✓" if rf_ev["schema_valid"] else "✗"
        log(f"    [response_format] {tag} {rf_out[:80]}")
        log(f"                       ({rf_elapsed:.2f}s)")

        all_results.append({
            "user": user,
            "raw": {"out": raw_out, "eval": raw_ev, "time": raw_elapsed},
            "rf":  {"out": rf_out, "eval": rf_ev, "time": rf_elapsed},
        })

    n = len(JSON_CASES)
    log("")
    log("=" * 72)
    log(f"  {n} 条测试结果汇总")
    log("=" * 72)
    log(f"{'指标':<24}{'裸 prompt':<18}{'response_format':<18}")
    log("-" * 60)
    for metric_name, key in [
        ("合法 JSON", "is_json"),
        ("字段齐全", "has_all_fields"),
        ("year 在 2015~2025", "year_in_range"),
        ("metric 在枚举内", "metric_in_enum"),
        ("jsonschema 完全通过", "schema_valid"),
    ]:
        rv = raw_stats[key]
        fv = rf_stats[key]
        log(f"{metric_name:<22}{rv}/{n} ({100*rv/n:.0f}%)      {fv}/{n} ({100*fv/n:.0f}%)")

    log("")
    log("结论: response_format 只保证是 JSON，不保证字段名、类型、枚举正确。")
    log("      guided_json（vLLM 私有）在解码层 100% 保证 schema 合法。")
    log("      这就是为什么生产环境 Agent 系统离不开约束解码。")
    return all_results


# ─────────────────────────────────────────────────────────
# Demo 4: response_format — OpenAI 标准方式
# ─────────────────────────────────────────────────────────
NEWS_SYSTEM = """你是新闻情感分析助手。分析用户给的新闻标题，输出 JSON 格式：
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0~1.0 的数值,
  "keywords": ["关键词1", "关键词2"]
}
不要输出任何其他文字。"""

NEWS_CASES = [
    "茅台三季度营收创历史新高，净利润同比增长 15%",
    "比亚迪召回 10 万辆电动车，涉及电池安全问题",
    "央行维持 LPR 利率不变",
    "宁德时代与宝马签订长期供货协议",
    "平安保险高管被调查，股价下跌 8%",
]


def eval_news(output):
    r = {"is_json": False, "has_sentiment": False,
         "valid_sentiment": False, "has_confidence": False, "has_keywords": False}
    try:
        obj = json.loads(output)
        r["is_json"] = True
    except json.JSONDecodeError:
        return r
    if "sentiment" in obj:
        r["has_sentiment"] = True
        if obj["sentiment"] in ("positive", "negative", "neutral"):
            r["valid_sentiment"] = True
    if "confidence" in obj and isinstance(obj["confidence"], (int, float)):
        r["has_confidence"] = True
    if "keywords" in obj and isinstance(obj["keywords"], list):
        r["has_keywords"] = True
    return r


def demo_response_format():
    section("Demo 4: response_format — OpenAI 标准 JSON 模式")
    log(f"Model: {MODEL}")

    raw_stats = {"is_json": 0, "has_sentiment": 0,
                 "valid_sentiment": 0, "has_confidence": 0, "has_keywords": 0}
    rf_stats  = {"is_json": 0, "has_sentiment": 0,
                 "valid_sentiment": 0, "has_confidence": 0, "has_keywords": 0}
    all_results = []

    for news in NEWS_CASES:
        log("")
        log(f"  ▶ {news}")
        msgs = [{"role": "system", "content": NEWS_SYSTEM},
                {"role": "user", "content": news}]

        # 裸 prompt
        t0 = time.time()
        raw_out = chat_complete(msgs, max_tokens=150)
        raw_elapsed = time.time() - t0
        raw_ev = eval_news(raw_out)
        for k in raw_stats:
            raw_stats[k] += int(raw_ev[k])
        tag = "✓" if raw_ev["is_json"] else "✗"
        log(f"    [裸 prompt     ] {tag} {raw_out[:100]}")
        log(f"                      ({raw_elapsed:.2f}s)")

        # response_format
        t0 = time.time()
        rf_out = chat_complete(msgs, max_tokens=150,
                               response_format={"type": "json_object"})
        rf_elapsed = time.time() - t0
        rf_ev = eval_news(rf_out)
        for k in rf_stats:
            rf_stats[k] += int(rf_ev[k])
        tag = "✓" if rf_ev["is_json"] else "✗"
        log(f"    [response_format] {tag} {rf_out[:100]}")
        log(f"                       ({rf_elapsed:.2f}s)")

        all_results.append({
            "news": news,
            "raw": {"out": raw_out, "eval": raw_ev, "time": raw_elapsed},
            "rf":  {"out": rf_out, "eval": rf_ev, "time": rf_elapsed},
        })

    n = len(NEWS_CASES)
    log("")
    log("=" * 72)
    log(f"  {n} 条测试结果")
    log("=" * 72)
    log(f"{'指标':<22}{'裸 prompt':<20}{'response_format':<18}")
    log("-" * 60)
    for name, key in [
        ("合法 JSON", "is_json"),
        ("有 sentiment 字段", "has_sentiment"),
        ("sentiment 值合法", "valid_sentiment"),
        ("有 confidence 字段", "has_confidence"),
        ("有 keywords 字段", "has_keywords"),
    ]:
        rv = raw_stats[key]
        fv = rf_stats[key]
        log(f"{name:<20}{rv}/{n} ({100*rv/n:.0f}%)      {fv}/{n} ({100*fv/n:.0f}%)")

    log("")
    log("观察: response_format 显著提升 JSON 合法率，但字段语义仍靠模型自觉。")
    log("      若需严格字段 schema，请用 guided_json（vLLM 私有扩展）。")
    return all_results


# ─────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────
def main():
    init_log()
    log("开始运行 vLLM Demo...")
    log(f"模型: {MODEL}")
    log(f"API:  {BASE_URL}")

    # 连接测试
    try:
        test_out = chat_complete(
            [{"role": "user", "content": "你好"}],
            max_tokens=10,
        )
        log(f"✓ API 连接成功: {test_out}")
    except Exception as e:
        log(f"✗ API 连接失败: {e}")
        log("请确保 Ollama 服务正在运行（ollama serve）")
        sys.exit(1)

    results = {}
    results["guided_choice"]   = demo_guided_choice()
    results["guided_regex"]    = demo_guided_regex()
    results["guided_json"]     = demo_guided_json()
    results["response_format"] = demo_response_format()

    section("运行完成")
    log("所有 Demo 运行完成。")
    log(f"日志已保存到: {LOG_FILE}")

    # 保存 JSON 结果
    json_file = os.path.join(OUTPUT_DIR, "demo_run_results.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"JSON 结果已保存到: {json_file}")

    print(f"\n全部完成！日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
