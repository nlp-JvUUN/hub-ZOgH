"""
llm_client.py — 统一 LLM 客户端（week15 编排系统用）

复用仓库根目录的 llm_config.py（曾文静/llm_config.py）：
  - 自动把「曾文静/」加入 sys.path，import llm_config
  - 统一读取 .env 里的 DEEPSEEK_API_KEY / base_url / model
  - 业务代码不散落任何 api_key 配置

只提供一个函数 llm_chat()：
  llm_chat(system, user, temperature=0.0, max_tokens=768, stop=["Observation:"])
  stop 参数用于 ReAct：让模型生成完 Action Input 就停，等工具结果续写。
"""

import os
import re
import sys
import time
from pathlib import Path

# ── 定位仓库根的 llm_config.py（曾文静/llm_config.py）──────────────
_HERE = Path(__file__).resolve()
for _cand in (_HERE.parents[2], _HERE.parents[1]):   # 曾文静/ → week15/
    if (_cand / "llm_config.py").exists():
        sys.path.insert(0, str(_cand))
        break
import llm_config as llm  # noqa: E402

# ── Mock 模式：未配置 API Key 时自动启用─────────────────────────────
# 用「脚本化大脑」+ 真实工具跑通全流程（天气 API / 文件读取都是真的），
# 便于离线自检与演示；配置好 .env 里的 DEEPSEEK_API_KEY 后自动切真实模型。
MOCK_MODE = not bool(llm.DEEPSEEK_API_KEY)

# 常见城市与 samples 目录文件（mock 大脑用它从问题里拆子任务）
_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "西安", "武汉"]
_FILES = ["notes_rag.md", "notes_agent.md", "notes_graph.md"]

# mock 模式模拟单次 LLM 调用的延迟（秒），让离线演示也能体现并行收益
MOCK_LLM_DELAY = float(os.getenv("WEEK15_MOCK_DELAY", "0.6"))


def _mock_llm(system: str, user: str) -> str:
    """脚本化大脑：按 agent 类型返回固定动作序列（两段式：先调工具，再作答）。
    用 history 里已有的 Observation 个数区分阶段（n_obs=0 先调工具，>0 作答）。"""
    n_obs = user.count("Observation:")
    if MOCK_LLM_DELAY > 0:
        time.sleep(MOCK_LLM_DELAY)   # 模拟真实 LLM 推理延迟
    if "任务编排主 Agent" in system:
        # 主 agent：第一轮派发，拿到 Observation 后综合
        if n_obs == 0:
            cities = [c for c in _CITIES if c in user]
            files = [f for f in _FILES if f in user]
            if len(cities) >= 2:
                spec = " | ".join(f"weather: {c}" for c in cities)
            elif len(files) >= 2:
                spec = " | ".join(
                    f"file: 用中文总结 data/{f} 并提炼 3 个核心要点" for f in files)
            elif cities:
                spec = f"weather: {cities[0]}"
            else:
                spec = f"file: 用中文总结 data/{files[0]} 并提炼 3 个核心要点" if files \
                    else "file: list_files"
            return (f"Thought: 任务可拆分为独立子任务，派发并行 worker\n"
                    f"Action: dispatch_workers\nAction Input: {spec}")
        # 拿工具结果后 → 模拟综合报告
        first_obs = user.split("Observation:", 1)[1].strip().splitlines()[0][:80]
        return (f"Thought: 已收齐所有 worker 结果\n"
                f"Final Answer: （mock 综合报告）{first_obs}；各 worker 详情见上方 Observation，"
                f"并行完成全部子任务。")
    if "城市天气调研员" in system:
        if n_obs == 0:
            city = re.search(r"Question: (.+)", user.split("Observation:")[0])
            task = city.group(1).strip() if city else ""
            city = next((c for c in _CITIES if c in task), "北京")
            return f"Thought: 查询 {city} 天气\nAction: city_weather\nAction Input: {city}"
        obs = user.rsplit("Observation:", 1)[1].strip().splitlines()[0][:60]
        return f"Thought: 已拿到天气数据\nFinal Answer: （mock）{obs}"
    if "文档加工员" in system:
        if n_obs == 0:
            return "Thought: 先查看可用文件\nAction: list_files\nAction Input: "
        if n_obs == 1:
            obs = user.rsplit("Observation:", 1)[1]
            fname = next((f for f in _FILES if f in obs), _FILES[0])
            return (f"Thought: 读取 {fname} 并加工\nAction: read_file\nAction Input: {fname}")
        obs = user.rsplit("Observation:", 1)[1].strip().splitlines()[0][:60]
        return (f"Thought: 已读完文件，完成加工\n"
                f"Final Answer: （mock 加工结果）{obs}")
    return "Thought: 直接作答\nFinal Answer: （mock）无可用脚本，请配置 API Key 后重试"


def llm_chat(system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 768, stop=None, retries: int = 3) -> str:
    """单轮 LLM 对话。失败自动重试（指数退避），与 ReAct 循环解耦。
    Mock 模式下返回脚本化回复（不消耗 API）。"""
    if MOCK_MODE:
        return _mock_llm(system, user)
    last_err = None
    for attempt in range(retries):
        try:
            return llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop or [],
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败（{retries} 次重试后）: {last_err}")


if __name__ == "__main__":
    print(llm_chat("你是测试助手", "请回复：配置验证成功"))
