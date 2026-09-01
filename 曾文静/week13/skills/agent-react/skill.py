"""agent-react — ReAct 循环元技能：把其他技能当作工具。

  - 这里 ReAct 循环是一个**普通技能**（agent-react），与其它技能同构：
    一样走 Lane 串行、一样产生事件流、一样写执行日志、一样受预算约束。
    外部通过统一消息协议调用它：{"skill": "agent-react", "inputs": {...}}。

大模型接入（复用根目录 llm_config.py 的统一 chat 接口）：
    每轮模型输出一个 JSON 对象，二选一：
      {"action": "call_tool", "thought": "...", "tool": "...", "params": {...}}
      {"action": "final_answer", "thought": "...", "answer": "..."}
"""

import json
import sys
from pathlib import Path

from skillflow.model import Progress

# 复用仓库根目录的 llm_config.py（老师统一配置模块：.env 存 Key + openai SDK +
# chat() 接口）。skill 里不硬编码任何 base_url / model / api_key。
# skill 模块可能被任意 cwd 加载，这里兜底把根目录加入 sys.path。
try:
    import llm_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # 曾文静/
    import llm_config


# ─────────────────────────────────────────────────────────────────────
# LLM 调用（可被 ctx.system["llm_client"] 替换，测试即用 mock）
# ─────────────────────────────────────────────────────────────────────


def default_llm_client(messages):
    """复用根目录 llm_config.py 的统一 chat 接口（低温度保证 JSON 输出稳定）。"""
    return llm_config.chat(messages, temperature=0.2)


def _extract_json(text: str):
    """从模型输出提取 JSON（兼容 ```json 包裹与前后杂质）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型输出中未找到合法 JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


# ─────────────────────────────────────────────────────────────────────
# 工具视图与系统提示词
# ─────────────────────────────────────────────────────────────────────


def _tool_view(spec_dict) -> str:
    """把一个技能规格压缩成给模型看的工具说明（只读 L1 元数据）。"""
    consumes = spec_dict.get("consumes", {})
    lines = []
    for pname, pinfo in consumes.items():
        req = "必填" if pinfo.get("required") else "可选"
        default = pinfo.get("default", "-")
        desc = pinfo.get("desc", "")
        lines.append(f"      - {pname} ({pinfo.get('type', 'str')}/{req}, 默认={default}): {desc}")
    params_text = "\n".join(lines) or "      （无参数）"
    return f"  - {spec_dict['name']}: {spec_dict.get('description', '')}\n    参数:\n{params_text}"


def _build_system_prompt(tool_views) -> str:
    return (
        "你是 SkillFlow 的调度助手，具备「思考 + 调用技能 + 观察」的 ReAct 能力。\n"
        "面对用户的问题：\n"
        "1. 先思考需要哪些信息或操作；\n"
        "2. 若需要调用技能，输出 call_tool 动作；\n"
        "3. 根据观察结果继续思考，可多次调用不同技能；\n"
        "4. 收集到足够信息后，输出 final_answer 给出最终回答。\n\n"
        f"可用技能列表：\n{chr(10).join(tool_views)}\n\n"
        "规则：\n"
        "- 只能调用列表中存在的技能，不得臆造。\n"
        "- 必填参数必须给出值；可选参数未明确时用默认值。\n"
        "- 数值参数必须是数字；内容参数直接采用用户原话中的值。\n"
        "- 技能执行失败时，观察结果会告诉你原因，你可以换技能、修正参数或直接说明。\n"
        "- 严格只输出一个 JSON 对象，不要 markdown 代码块、不要多余解释。\n\n"
        "输出格式二选一：\n"
        '调用技能: {"action": "call_tool", "thought": "<简短思考>", "tool": "<技能名>", "params": {...}}\n'
        '最终回答: {"action": "final_answer", "thought": "<简短思考>", "answer": "<给用户的最终回答>"}'
    )


# ─────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────


def _execute_tool(executor, tool: str, params: dict):
    """执行一个技能，任何失败都转成可读字符串（作为观察回喂模型）。"""
    if executor is None:
        return False, "未注入技能执行器（execute_skill 服务缺失）"
    try:
        result = executor(tool, params)
        return True, json.dumps(result, ensure_ascii=False)[:2000]
    except Exception as e:
        return False, f"技能执行失败: {type(e).__name__}: {e}"


def run(ctx, question: str, max_iterations: int = 6, **inputs):
    max_iterations = max(1, int(max_iterations))

    list_skills = ctx.system.get("list_skills")
    if list_skills is None:
        raise RuntimeError("agent-react 缺少 list_skills 系统服务")
    tool_views = [
        _tool_view(s)
        for s in list_skills()
        if s.get("name") != ctx.spec.name  # 禁止递归调用自身
    ]
    if not tool_views:
        raise RuntimeError("没有可用的其他技能可供调度")

    executor = ctx.system.get("execute_skill")
    llm = ctx.system.get("llm_client") or default_llm_client

    messages = [
        {"role": "system", "content": _build_system_prompt(tool_views)},
        {"role": "user", "content": question},
    ]
    steps = []

    for i in range(max_iterations):
        yield Progress(done=i + 1, total=max_iterations, message=f"ReAct 第 {i + 1}/{max_iterations} 轮推理…")

        raw = llm(messages)
        parsed = _extract_json(raw)
        action = parsed.get("action")
        thought = parsed.get("thought", "")

        if action == "final_answer":
            return {
                "answer": parsed.get("answer", ""),
                "steps": steps,
                "iterations": i + 1,
            }

        if action == "call_tool":
            tool = parsed.get("tool")
            params = parsed.get("params", {})
            steps.append({"round": i + 1, "action": "call_tool", "thought": thought, "tool": tool, "params": params})
            ok, obs = _execute_tool(executor, tool, params)
            steps.append({"round": i + 1, "action": "observation", "success": ok, "observation": obs})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"观察结果: {obs}"})
            continue

        # 无法识别的动作：把纠正信息回喂给模型，让其重试
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"无法识别的 action: {action!r}，请输出 call_tool 或 final_answer。"})

    return {
        "answer": f"已达到最大推理轮数（{max_iterations}），未能给出最终答案。",
        "steps": steps,
        "iterations": max_iterations,
    }
