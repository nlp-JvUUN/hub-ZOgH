"""
作业（LangChain 版）：主 Agent 下发多个带 Tavily 搜索工具的子 Agent，并行完成检索调研任务

架构：
        主 Agent (Coordinator，手写调度逻辑)
        ├─ ① 任务分解：LangChain chain 输出 JSON 子任务清单
        ├─ ② 并行下发：asyncio.gather 并发跑 N 个 LangChain 工具调用子 Agent，
        │            每个子 Agent 自带一把 Tavily 搜索工具（create_agent）
        └─ ③ 汇总：LangChain chain 把子结果整理成最终报告

依赖（装进你的 conda 环境 LangGraph，激活后运行）：
  conda activate LangGraph
  pip install langchain langchain-openai langchain-community langchain-tavily python-dotenv

密钥（写在同目录 .env 文件里，脚本启动时自动读取）：
  DEEPSEEK_API_KEY   在 https://platform.deepseek.com 申请
  TAVILY_API_KEY     在 https://app.tavily.com 免费申请（每月约 1000 次额度）

运行（务必在 conda LangGraph 环境里）：
  conda activate LangGraph
  python agent_langchain.py

说明：
  - 编排逻辑（分解→下发→汇总）是手写的，这是作业要求的"自己实现"部分；
    只有子 Agent 内部的"思考-搜索-作答"循环借用了 LangChain 的 create_agent。
  - 并行 = asyncio.gather：N 个搜索子任务同时发起，总耗时 ≈ 最慢那个子任务。
"""

import asyncio
import json
import os
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 新版官方 Tavily 集成优先；找不到就退回 langchain-community（两者调用方式兼容）
try:
    from langchain_tavily import TavilySearch as SearchTool
    print("[加载] langchain_tavily.TavilySearch")
except ImportError:
    from langchain_community.tools import TavilySearchResults as SearchTool
    print("[加载] langchain_community.TavilySearchResults")

# ---------------- 配置 ----------------
# 自动读取脚本同目录下的 .env 文件，把 DEEPSEEK_API_KEY / TAVILY_API_KEY 填进环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY", "在这里填你的key"),
    temperature=0.2,
)

MISSION = "调研 2026 年大模型智能体（AI Agent）领域最重要的 3 个技术趋势，每个趋势给出具体案例与来源。"


# ---------------- ① 主 Agent：任务分解 ----------------
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是主 Agent（Coordinator）。把用户的大任务拆成 2~5 个相互独立、可并行执行的检索子任务。"
               "只输出一个 JSON 数组，不要任何其他内容：\n"
               '[{{"id":1,"name":"子任务名","desc":"给子agent的详细指令"}}, ...]'),
    ("human", "大任务：{mission}"),
])
decompose_chain = DECOMPOSE_PROMPT | llm | StrOutputParser()


# ---------------- ② 子 Agent：LangChain 工具调用 Agent + Tavily 搜索 ----------------
SUBAGENT_SYSTEM = """你是主 Agent 派发的子 Agent（检索调研员），只负责完成分配给你的那一个子任务。
你可以使用 Tavily 搜索工具获取实时信息，搜索后再作答，并注明信息来源。"""


def make_subagent():
    """构造一个带 Tavily 搜索工具的子 Agent（LangChain 1.x create_agent，工具调用型）。"""
    search = SearchTool(max_results=5)
    return create_agent(model=llm, tools=[search], system_prompt=SUBAGENT_SYSTEM)


async def subagent(task: dict) -> dict:
    print(f"  [子Agent {task['id']}] 开始：{task['name']}")
    agent = make_subagent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task["desc"]}]})
    content = result["messages"][-1].content
    if isinstance(content, list):                  # 个别模型返回的是内容块列表，拼成字符串
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    print(f"  [子Agent {task['id']}] 完成：{task['name']}")
    return {"id": task["id"], "name": task["name"], "result": content}


# ---------------- ③ 主 Agent：汇总 ----------------
AGGREGATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是主 Agent。下面是你派发出去的所有子 Agent 的调研结果，"
               "请整理成一份逻辑清晰、结构完整的最终报告，保留信息来源。"),
    ("human", "原任务：{mission}\n\n各子任务结果：\n{results}"),
])
aggregate_chain = AGGREGATE_PROMPT | llm | StrOutputParser()


# ---------------- 主流程 ----------------
async def run():
    t0 = time.perf_counter()

    print("① 主 Agent 任务分解 ...")
    text = (await decompose_chain.ainvoke({"mission": MISSION})).strip()
    if text.startswith("```"):                    # 去掉模型可能加的 ```json 代码块
        text = text.split("```")[1]
    subtasks = json.loads(text)
    print(f"   共 {len(subtasks)} 个子任务：{[t['name'] for t in subtasks]}")

    print("② 并行下发子 Agent ...")
    results = await asyncio.gather(*(subagent(t) for t in subtasks))

    print("③ 主 Agent 汇总结果 ...")
    parts = "\n\n".join(f"### 子任务 {r['id']}：{r['name']}\n{r['result']}" for r in results)
    final = await aggregate_chain.ainvoke({"mission": MISSION, "results": parts})

    print(f"\n总耗时 {time.perf_counter() - t0:.1f} 秒")
    print("=" * 70)
    print("最终报告：")
    print("=" * 70)
    print(final)


if __name__ == "__main__":
    asyncio.run(run())
