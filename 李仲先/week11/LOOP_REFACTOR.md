# 天气查询工具调用改造为循环调用 — 改动总结

## 改造目标

将天气查询（及全部工具调用）从**单轮闭环**改为**多轮 ReAct 循环**，使模型能根据工具返回结果决定下一步操作，而非仅在一轮内完成所有工具调用。

### 单轮 vs 多轮对比

| | 单轮闭环（改造前） | 多轮循环（改造后） |
|---|---|---|
| 流程 | 提问 → 工具调用（一轮）→ 最终回答 | 提问 → 工具调用 → 看结果 → 再调工具 → ... → 最终回答 |
| 工具调用 | 仅一轮，可并行多工具 | 多轮，每轮可串行/并行，根据前轮结果决策 |
| 适用场景 | 工具参数已知、无依赖关系 | 工具间有依赖（如：查A结果→决定查B） |
| 新增文件 | 无 | `run_*_loop.py` × 3 |

---

## 改动内容

### 新增文件

| 文件 | 说明 |
|---|---|
| `mode_function_call/run_function_call_loop.py` | 方式一（Function Call）多轮循环版 |
| `mode_mcp/run_mcp_loop.py` | 方式二（MCP）多轮循环版 |
| `mode_cli/run_cli_loop.py` | 方式三（CLI）多轮循环版 |

### 核心改动：`run()` → `run_loop()`

以 `run_function_call_loop.py` 为例，循环逻辑如下：

```python
def run_loop(client, model, question, max_rounds=5):
    messages = [system_prompt, user_question]
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        resp = client.chat.completions.create(
            model=model, messages=messages,
            tools=TOOLS_SCHEMA, tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型不再调用工具 → 已有最终回答，结束循环
        if not msg.tool_calls:
            return {"answer": msg.content, "rounds": rounds, ...}

        # 模型调用了工具 → 执行、回填，继续循环
        messages.append(msg)
        for tc in msg.tool_calls:
            result = TOOL_DISPATCH[tc.function.name](**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # 达到 max_rounds 仍未结束 → 强制生成回答
    return {"answer": ..., "rounds": max_rounds}
```

**循环终止条件**：
1. 模型输出的 `message` 不含 `tool_calls` → 认为已给出最终回答
2. 达到 `max_rounds`（默认5，可通过 `--loop` 参数调整）

### 三方式循环版差异点

| | Function Call 循环版 | MCP 循环版 | CLI 循环版 |
|---|---|---|---|
| 工具执行 | 直接调 `TOOL_DISPATCH[name](**args)` | `await session.call_tool(name, args)` | `executor(args)`（subprocess） |
| 异步 | 同步 | `async/await` | 同步 |
| 工具发现 | 手写 `TOOLS_SCHEMA` | `list_tools()` 一次发现，后续复用 | 手写 `TOOLS_SCHEMA` |
| 新增参数 | `--loop N`（最大轮次） | `--loop N` | `--loop N` |

### 系统提示词改动

所有循环版的系统提示词新增了多轮能力说明：

```
你可以多次调用工具来获取信息，然后基于工具返回的结果
决定是否需要调用更多工具，直到你收集到足够信息再给出最终回答。
```

---

## 演示问题（内置 `--demo`）

循环版新增了依赖工具结果的演示问题：

```python
DEMO_QUESTIONS = [
    "北京天气如何？如果下雨请再查上海的天气做对比。",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比北京、上海、广州三座城市的天气。",
    "比亚迪2023年营收是多少？",  # 幻觉控制
]
```

第一个问题特意设计为**条件依赖**：模型需要先查北京天气，看到结果后才能决定是否查上海。

---

## 运行示例

```bash
# 方式一循环版
python mode_function_call/run_function_call_loop.py --demo
python mode_function_call/run_function_call_loop.py -q "北京天气如何？如果下雨请再查上海" --loop 3

# 方式二循环版
python mode_mcp/run_mcp_loop.py --demo

# 方式三循环版（named 形态）
python mode_cli/run_cli_loop.py --mode named --demo

# 方式三循环版（bash 形态）
python mode_cli/run_cli_loop.py --mode bash -q "北京天气如何？如果下雨请再查上海"
```

---

## 与原有单轮版的关系

- **原有文件不变**：`run_function_call.py`、`run_mcp.py`、`run_cli.py` 保留，继续用于教学对比
- **循环版是扩展**：展示从单轮到多轮的演进，帮助学生理解 ReAct 模式
- **`compare.py` 暂不支持循环版**：对比运行器仍使用单轮版，循环版需单独运行

---

## 已知限制

1. **循环次数控制**：`max_rounds` 是硬上限，达到后强制结束。若模型仍需要更多轮次，需调大 `--loop` 参数
2. **无中间回答**：循环过程中模型不输出中间思考，只在最后给出回答（可扩展为流式输出每轮）
3. **DeepSeek 偶尔不返回 tool_calls**：与单轮版相同的问题，若频繁出现可换 `--provider dashscope`
4. **MCP 循环版 Server 生命周期**：`AsyncExitStack` 在 `main_async` 内管理，整个循环过程 Server 保持连接，无需重连

---

## 改动文件清单

```
新增：
  mode_function_call/run_function_call_loop.py
  mode_mcp/run_mcp_loop.py
  mode_cli/run_cli_loop.py

修改：
  AGENTS.md  （新增循环版命令、文件列表、架构说明）
```
