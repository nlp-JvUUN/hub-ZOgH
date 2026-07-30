---
name: agent-react
version: 1.0.0
description: >-
  以 ReAct 循环把其他技能当作工具使用的元技能：大模型思考 → 调用技能 → 观察结果 → 再思考，
  直到给出最终回答。这是课件「ReAct 循环 + SKILL.md 注入」在 harness 内的落地形态。
weight: 1
consumes:
  question:
    type: str
    required: true
    desc: 用户的自然语言问题
  max_iterations:
    type: int
    required: false
    default: 6
    desc: 最大推理轮数（防死循环）
provides:
  answer: 最终回答
  steps: 推理步骤（思考/调用/观察）
  iterations: 实际轮数
deps: []
heartbeat: null
tags: [agent, llm, react]
---

# ReAct 元技能

与「把 ReAct 循环写在 harness 之外的旁路模块」不同，本技能把循环本身做进
harness：它与其他技能完全同构 —— 受 Lane 串行、执行日志、加载预算约束，
外部调用它的方式与调用普通技能完全一致（`{"skill": "agent-react", ...}`）。

## 运行原理

1. harness 通过 ctx.system 注入两个服务：
   - `list_skills()`：L1 元数据视图（不加载任何实现），用作工具清单；
   - `execute_skill(name, params)`：执行另一个技能并返回其输出（含错误防护）。
2. 每轮：调用大模型（OpenAI 兼容 API，urllib 实现，环境变量配置）→
   解析 JSON 动作（`call_tool` / `final_answer`）→ 执行或回答。
3. 工具执行失败/技能不存在会作为「观察」回喂给模型，让其换工具或修正参数
   —— 而不是直接崩溃（错误恢复）。
4. 每轮 yield `Progress`，ReAct 推理过程对 CLI / SSE 渐进可见。

## LLM 配置（复用根目录 llm_config.py）

大模型接入不在这里重复配置：直接复用仓库根目录的 `llm_config.py`
（老师统一配置模块 —— `.env` 存 Key + openai SDK + `chat()` 接口），
skill 里不硬编码任何 base_url / model / api_key。

- 配置 Key：在根目录 `.env` 填入 `DEEPSEEK_API_KEY=sk-xxx`（模板见 `.env.example`）
- 换模型：设置环境变量 `LLM_MODEL` / `DEEPSEEK_MODEL`，或改 `llm_config.DEFAULT_MODEL`
- 未配置 Key 时，`llm_config` 会给出明确报错（提示去 .env 填 Key）

测试可用 `ctx.system["llm_client"]` 注入 mock（不依赖真实 API）。
