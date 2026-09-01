"""
run_demo.py — 多轮对话能力验证剧本（自动跑完整场景并输出验证记录）

三个剧本分别对应多轮对话的三种关键能力：

  A. 追问复用（指代消解）：同一会话内连续追问，坐标/城市从记忆复用，
     不重复 geocode —— 工具调用数逐轮下降，直接证明「记忆生效」；
  B. 纯记忆回答：前两轮把两城温度存进事实表，第三轮「哪个更冷」
     不调用任何工具，直接从事实回答 —— 证明「事实层可被模型消费」；
  C. 长对话压缩：连续 20 个问题把窗口挤爆，滚动摘要被反复触发，
     上下文 token 被预算钳制 —— 证明「对话可以无限长而 token 不爆炸」。

运行方式：
  # 离线（mock 驱动 + 模拟城市库，零 API 成本）—— 默认
  python run_demo.py

  # 真实模型（DEEPSEEK_API_KEY + Open-Meteo 免费天气接口）
  python run_demo.py --real

  # 输出 Markdown 验证记录（供 docs/ 存档 / 提交说明）
  python run_demo.py --out docs/验证记录_mock离线.md

设计差异（对照同学作业）：
  大多数同学用「真实模型跑一遍人工对话」做验证，肉眼看不出来记忆到底存了什么；
  本脚本在每个场景的关键轮次打印【记忆内部状态】（事实表 / 摘要 / token 统计），
  并给出每个场景的「通过标准」自动判定 —— 记忆层是否生效是可验证的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chat_agent import build_agent  # noqa: E402
from memory import MemoryManager  # noqa: E402

# ── 剧本定义 ────────────────────────────────────────────────────────────────

SCENARIO_A = [
    "宁德今天天气怎么样？",
    "那空气质量呢？",            # 指代消解：复用宁德坐标，不应再 geocode
    "明天呢？",                  # 指代消解：复用宁德坐标，直接 forecast
]

SCENARIO_B = [
    "北京今天天气怎么样？",
    "上海呢？",                  # 追问：查上海（北京仍在记忆里）
    "北京和上海今天哪个更冷？",   # 纯记忆回答：两城温度都在事实表 → 0 工具调用
]

SCENARIO_C = [  # 长对话压缩：同城多指标反复查，把窗口与预算挤爆
    "广州今天天气怎么样？",
    "深圳今天天气怎么样？",
    "广州的空气质量呢？",
    "深圳的空气质量呢？",
    "广州未来3天预报？",
    "深圳未来3天预报？",
    "广州今天气温多少度？",
    "深圳今天气温多少度？",
    "广州现在湿度大吗？",
    "深圳现在湿度大吗？",
    "广州明天会下雨吗？",
    "深圳明天会下雨吗？",
    "广州AQI是多少？",
    "深圳AQI是多少？",
    "广州这几天适合户外运动吗？",
    "深圳这几天适合户外运动吗？",
    "广州和深圳哪个城市今天更热？",
    "广州和深圳哪个空气质量更好？",
    "广州现在风大吗？",
    "深圳现在风大吗？",
]

SCENARIOS = [
    ("A · 追问复用（指代消解）", SCENARIO_A,
     "后续轮工具调用数应比首轮少（复用坐标事实，跳过 geocode）"),
    ("B · 纯记忆回答（事实层消费）", SCENARIO_B,
     "第三轮工具调用数应为 0（两城温度均来自事实表）"),
    ("C · 长对话压缩（滚动摘要 + token 预算）", SCENARIO_C,
     "20 轮后记忆 turns ≤ 窗口，摘要非空，上下文 token 受预算钳制"),
]


def run_scenario(agent, title: str, questions: list, check: str, out):
    print("\n" + "=" * 70)
    print(f"场景 {title}")
    print(f"通过标准：{check}")
    print("=" * 70)
    results = []
    for q in questions:
        final = None
        for step in agent.chat(q):
            if step["type"] == "action" and step.get("observation", "").startswith("[ERROR]"):
                print(f"  ⚠️  工具报错: {step['observation'][:80]}")
            if step["type"] == "final":
                final = step
        assert final is not None, "chat() 必须产出 final 步"
        u = final.get("usage", {}) or {}
        tool_count = len(final.get("turn", {}).get("tools", []))
        mem = final.get("memory", {})
        results.append({"q": q, "tools": tool_count,
                        "answer": final["answer"][:120]})
        print(f"\n  你: {q}")
        print(f"  ✅ {final['answer'][:200]}")
        print(f"  ─ 工具 {tool_count} 次 | token {u.get('total_tokens', '-')} | "
              f"记忆 {mem.get('turns', '-')} 轮/事实 {mem.get('facts', '-')} 条/"
              f"摘要 {mem.get('summary_tokens', '-')} tok")

    # 自动判定（mock 为确定性判定依据；真实模型结果供参考）
    verdicts = []
    if title.startswith("A"):
        verdicts.append(("追问轮工具数下降", results[1]["tools"] < results[0]["tools"] and
                         results[2]["tools"] < results[0]["tools"]))
        verdicts.append(("追问轮仅 1 次调用（坐标复用，跳过 geocode）",
                         results[1]["tools"] == 1 and results[2]["tools"] == 1))
    elif title.startswith("B"):
        verdicts.append(("对比轮 0 工具调用", results[2]["tools"] == 0))
    elif title.startswith("C"):
        m: MemoryManager = agent.memory
        verdicts.append(("窗口未失控", m.stats()["turns"] <= m.window_turns + 2))
        verdicts.append(("摘要已生成", len(m.summary) > 0))
        verdicts.append(("token 受控", m.stats()["context_tokens"] < 4000))
    for name, ok in verdicts:
        print(f"  {'✅' if ok else '❌'} 判定[{name}]")

    out.append(f"\n### 场景 {title}\n")
    out.append(f"通过标准：{check}\n")
    for i, r in enumerate(results, 1):
        out.append(f"**第 {i} 问**：{r['q']}  \n工具调用 {r['tools']} 次  \n> {r['answer']}\n")
    if verdicts:
        out.append("判定：" + "，".join(f"{'✅' if ok else '❌'} {n}" for n, ok in verdicts) + "\n")

    # 记录场景结束时的记忆内部状态（摘要/事实），便于核对"记忆里到底存了什么"
    m: MemoryManager = agent.memory
    out.append(f"**场景结束时记忆状态**：正文 {m.stats()['turns']} 轮（窗口 {m.window_turns}），"
               f"事实 {len(m.facts)} 条，摘要约 {m.stats()['summary_tokens']} tok\n")
    if m.facts:
        out.append("关键事实：\n" + "\n".join(f"- {f}" for f in m.facts) + "\n")
    if m.summary:
        out.append(f"滚动摘要：\n> {m.summary}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真实模型（默认 mock）")
    ap.add_argument("--mock-tools", dest="mock_tools", action="store_true", default=None,
                    help="强制使用离线工具后端（真实模型模式也可用，验证更可复现）")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default=None, help="Markdown 验证记录输出路径")
    ap.add_argument("--scenarios", default="A,B,C",
                    help="运行哪些场景（逗号分隔，默认 A,B,C）")
    args = ap.parse_args()

    # --real 未显式指定 --mock-tools 时：真实模型走真实工具；mock 模式默认离线工具
    use_mock_tools = args.mock_tools if args.mock_tools is not None else (not args.real)

    agent = build_agent(provider=args.provider, model=args.model, mock=not args.real,
                        max_steps=8, window_turns=6, token_budget=4000,
                        mock_tools=use_mock_tools)
    tool_mode = "离线工具" if use_mock_tools else "真实工具(Open-Meteo)"
    print(f"驱动: {'真实模型' if args.real else 'mock（离线）'} | 工具: {tool_mode} | "
          f"窗口 {agent.memory.window_turns} 轮 | 预算 {agent.memory.token_budget} tok\n")

    out = []
    out.append(f"# 多轮对话验证记录（{'真实模型' if args.real else 'mock 离线'} × {tool_mode}）\n")
    out.append(f"窗口 {agent.memory.window_turns} 轮 · 预算 {agent.memory.token_budget} tok · "
               f"最大步数 {agent.max_steps}\n")
    wanted = {s.strip().upper() for s in args.scenarios.split(",") if s.strip()}
    for title, questions, check in SCENARIOS:
        if not any(title.split("·")[0].strip().startswith(w) for w in wanted):
            continue
        run_scenario(agent, title, questions, check, out)
        agent.memory.reset()  # 场景之间隔离记忆

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(out), encoding="utf-8")
        print(f"\n验证记录已写入 {args.out}")


if __name__ == "__main__":
    main()
