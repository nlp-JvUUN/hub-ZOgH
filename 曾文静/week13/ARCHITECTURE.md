# SkillFlow 技术架构

## 一、设计哲学：把「渐进式」变成可验证的机制

课件把 Harness Engineering 定义为「模型在什么机制里运行」，并给出了
Fat Gateway、Lane 队列、四层记忆、Memory Flush、HEARTBEAT、Skills 等概念。
本作业用一套技能执行框架把这些概念**收敛成可运行的机制**，
并让「渐进式加载执行」落在三条主轴上：

```
                加载渐进（Load）              执行渐进（Run）               交付渐进（Deliver）
              ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────────┐
  请求 ──▶ L0 列目录名            │      │ stage 1: 生成器   │      │ 事件总线 Event[]      │
              L1 SKILL.md 元数据  │ ───▶ │ stage 2: 契约注入 │ ───▶ │ CLI 逐条打印           │
              L2 skill.py 实现    │      │ stage 3: 流式产出 │      │ HTTP-SSE 实时推送      │
              L3 resources/ 资源  │      │ stage 4: 失败降级 │      │ Markdown 记忆日志      │
              （只解析变化的部分）  │      │ （部分结果不丢）  │      │ HEARTBEAT 主动触发     │
              └──────────────────┘      └──────────────────┘      └──────────────────────┘
```

## 二、模块图

```
                         ┌─────────────────────────── HarnessApp（组装层）───────────────────────────┐
                         │  skills_dir ──▶ Manifest ──▶ Registry（L1 只读视图）                       │
                         │      │                          │                                          │
                         │      ▼                          ▼                                          │
                         │  SkillRuntime（L2/L3 懒加载 + LoadBudget）──▶ PipelineEngine（执行引擎）    │
                         │                                                     │                      │
                         │  Journal（MD 日志 + Memory Flush）◀── 事件 ──────────┤ 生成器事件流          │
                         │      ▲                                               │                      │
                         │  HeartbeatScheduler ──▶ SessionHub（Lane 队列）◀── 投递消息                  │
                         └──────┼───────────────────────────┬───────────────────┘                      │
                                │                           │                                          │
                          CLI / REPL                  HTTP 网关（SSE / 轮询）                           │
```

- **Manifest（discovery.py）**：L0/L1 层。`scan()` 按 `(md5, mtime)` 指纹做增量扫描，
  未变化的 skill 命中缓存、零解析成本；`watch()` 轮询增量扫描，是「运行中热添加」的入口。
- **SkillRuntime（loader.py）**：L2/L3 层。实现文件只在首次执行时 `importlib` 加载；
  实现文件被修改后自动重载（热更新）。`LoadBudget` 限制已加载 weight 总量，
  超预算的 skill 由引擎发出 `stage_defer` 事件，不执行、不报错。
- **PipelineEngine（engine.py）**：执行层。`run()` 本身是**生成器**，逐个产出事件；
  先做依赖展开（DFS 后序 + 环检测），再逐 stage：加载 → 契约注入 → 校验 → 流式执行。
- **SessionHub（session.py）**：会话层。`InternalMessage` 归一化消息（Channel Adapter 思想）；
  每个会话一条 Lane（deque + 工作线程），消息严格 FIFO 串行；
  三标志 `is_running / has_error / retry_count`：失败自动重试同一消息，
  超限则 `has_error + paused`，需 `resume()` 确认后继续（课件 slide 11 的语义）。
- **Journal（journal.py）**：记忆层。每条事件实时追加 `journal/YYYY-MM-DD.md`（录音）
  与 `events.jsonl`；`flush()` 把当天日志聚合摘要写入 `journal/MEMORY.md`（纪要），
  即规则式 Memory Flush（接口留了 `summarizer` 参数，可换成 LLM 提炼）。
- **HeartbeatScheduler（scheduler.py）**：调度层。技能在 SKILL.md 声明
  `heartbeat: 30s / 5m / 1h / daily 23:59`，到点把消息投进 `__heartbeat__` 会话的
  Lane —— 心跳任务与用户消息走**同一条**执行路径（课件 HEARTBEAT.md 概念）。

## 三、Skill 契约（SKILL.md frontmatter）

```yaml
---
name: word-count              # 唯一名（目录名）
version: 1.0.0
description: 统计文本的单词数   # 人类/LLM 可读的功能说明
weight: 1                     # L2 加载代价，配合 LoadBudget 做加载预算
consumes:                     # 输入契约（管道对接的钥匙）
  text:
    type: str
    required: true
    desc: 要统计的文本
  top_n:
    type: int
    required: false
    default: 5
provides:                     # 输出契约
  count: 单词总数
deps: []                      # 依赖的其他 skill（执行前自动先跑）
heartbeat: null               # 心跳周期；非空则被 HeartbeatScheduler 调度
tags: [demo, text]
---
# 正文：给 LLM / 人类看的执行说明（可注入 prompt）
```

实现（`skill.py`）三种写法，引擎全部兼容：

```python
# 1) 生成器流式（推荐：渐进式输出）
def run(ctx, text: str, **inputs):
    for i in range(3):
        yield Progress(done=i + 1, total=3, message=f"第 {i+1}/3 步")
    return {"count": len(text.split())}     # return 值 = 输出

# 2) 普通函数：直接 return dict
def run(ctx, text: str, **inputs): return {"count": len(text.split())}

# 3) 类式
class Skill:
    def run(self, ctx, **inputs): return {...}
```

`ctx`（StageContext）提供：`ctx.inputs`（已解析输入）、`ctx.spec`、`ctx.resources()` /
`ctx.resource(name)`（L3 资源按需读取，读动作产生 `load(L3)` 事件）、
`ctx.system`（harness 注入的系统服务信息，如 journal 目录）。

## 三·五、元技能：ReAct 循环本身是一个技能（agent-react）

```
POST /api/sessions/{sid}/messages  {"skill": "agent-react", "inputs": {"question": "..."}}
```

HarnessApp 通过 `ctx.system` 注入两个服务（`app.py`）：

| 服务 | 作用 |
|------|------|
| `list_skills()` | L1 元数据视图（不加载任何实现），生成给模型看的工具清单 |
| `execute_skill(name, params)` | 受控执行另一个技能并返回输出；禁止递归调用 agent-react |

每轮推理：模型输出 JSON 动作（`call_tool` / `final_answer`）→ 执行或回答；
技能不存在 / 执行失败会作为「观察」回喂给模型恢复；每轮 yield `Progress`，
ReAct 推理过程对 CLI / SSE 渐进可见；`max_iterations` 防死循环。
LLM 接入复用仓库根目录的 `llm_config.py`（老师统一配置模块：`.env` 存
`DEEPSEEK_API_KEY`、openai SDK、`chat()` 接口），skill 里不硬编码任何
base_url / model / api_key；测试可通过 `ctx.system["llm_client"]` 注入 mock（见 tests）。

## 四、事件协议（渐进式执行对外的统一语言）

| kind | 含义 | payload |
|------|------|---------|
| `discover` | 管道规划结果 | order / budget |
| `load` | L2/L3 加载 | stage / resource / budget |
| `progress` | 生成器进度 | done / total / percent / message |
| `stage_ok` | 单 stage 成功 | output / duration_ms |
| `stage_fail` / `stage_skip` / `stage_defer` | 失败 / 级联跳过 / 预算推迟 | error / reason |
| `report` | 管道汇总 | status / stages / duration_ms / message |

CLI 把事件渲染成图标行；HTTP 网关把事件序列化成 SSE `data:` 帧；
Journal 把事件落成 Markdown —— **同一条事件流，三种消费方式**。

## 五、与课件 week13 概念的对应

| 课件概念 | 本作业落点 |
|---------|-----------|
| Fat Gateway / Channel Adapter | `SessionHub.submit()` 统一消息入口，`InternalMessage` 归一化 |
| Lane 队列（isRunning/hasError/retryCount） | `Session` 的 lane + 三标志 + resume 确认 |
| 四层记忆（Working→Short→Long→Vector） | `Event`（工作）→ 每日日志（短期）→ MEMORY.md（长期），检索留给上层 |
| Memory Flush | `Journal.flush()`，由 daily-report 心跳技能定时触发 |
| HEARTBEAT.md | `HeartbeatScheduler` + SKILL.md 的 `heartbeat` 声明 |
| Skills（SKILL.md） | 每个 skill 目录 = SKILL.md（元数据）+ skill.py（实现）+ resources/（资源） |
| Context Engine / 上下文组装 | `StageContext`：契约注入 + 系统服务注入 + 参数校验 |
