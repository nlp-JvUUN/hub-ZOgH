# ARCHITECTURE.md — 技能注册表驱动的 Subagent 并行编排系统

## 1. 项目定位

**场景**：用户提出一个任务（对比多个城市天气 / 加工多份文档 / 多对象调研…），
主编排 agent 自主决定是否派发多个 worker 并行执行，收齐后综合成最终答案。

**核心设计**：
- 主 agent **不直接持有任何业务工具**，只有 2 个编排工具：
  - `list_skills`：查看可派发的 worker 技能清单（L1 元数据视图，不加载实现）；
  - `dispatch_workers`：按「技能名: 任务 | 技能名: 任务」派发并行 worker。
- 业务能力全部下沉到**技能注册表**（`skills.py`），每个技能声明 名字/别名/描述/工具集/提示词；
  主 agent 按技能名构造 worker —— **编排引擎与具体场景解耦**，加场景 = 注册表加一条。
- worker 也是 ReAct 循环（复用同一个 `ReActLoop` 类），区别只在工具集与提示词；
  N 个 worker 用 `ThreadPoolExecutor` 并行执行 → 并行优势。
- **schema-first 交接**：派发参数解析成结构化任务契约 `{skill, task}`；
  worker 结果以 `{node_id, skill, status, final_answer, trace, duration}` 契约回收，
  汇总时才格式化成主 agent 的 Observation 文本。
- **节点级可观测**：一次运行生成一个 `graph_id`，主/worker 每条 trace 都带
  graph_id + node_id + 耗时（PPT 落地要点），CLI 输出拓扑图。

**范式归属**：动态 Orchestrator-Workers（PPT 6.3）——主 agent 决定派几个、用什么技能，
拓扑在运行时生长；工具调用的 fan-out/fan-in 对应 PPT 6.3 的 Diamond 结构要素。

## 2. 整体流水线

```
用户问题
   ↓
主编排 Agent ReAct（工具: list_skills + dispatch_workers）
   ├─ 可拆分任务 → dispatch_workers("weather: 北京 | weather: 上海 | file: 总结...")
   │                  ↓ fan-out（ThreadPoolExecutor 并行）
   │         ┌─ worker1 ReAct(weather: city_weather) ─┐
   │         ├─ worker2 ReAct(weather: city_weather) ─┤ 并行
   │         └─ worker3 ReAct(file: read_file/list_files) ─┘
   │                  ↓ fan-in（契约回收 + 截短）
   └─ 综合成最终报告（含并行加速统计）→ Final Answer
```

脚本对应：`llm_client.py`(统一客户端/mock) → `react_loop.py`(通用 ReAct) →
`skills.py`(技能注册表) → `dispatch.py`(派发引擎) → `main_agent.py`(主编排) →
`demo.py`(CLI) / `eval_compare.py`(A/B)。

## 3. 各环节技术选型

### 3.1 统一 LLM 客户端（llm_client.py）
复用仓库根 `llm_config.py`（自动把 `曾文静/` 加入 sys.path），业务代码零散配置。
**Mock 模式**：未配置 API Key 时自动启用「脚本化大脑 + 真实工具」——
天气 API 真实调用、文件真实读取，全流程离线可跑通；配置 Key 后自动切真实模型。

### 3.2 通用 ReAct 引擎（react_loop.py）
主 agent 与 worker 共用 `ReActLoop`，区别只在 `tools` 与 `system_prompt`。
- 经典 ReAct：LLM 输出 Thought/Action/Action Input，`stop=["Observation:"]` 截断，
  工具结果作为 Observation 续写，直到 Final Answer；
- **Action Input 跨行解析**：直到下一个标记或文本结束，并剔除模型补的括号式旁白行；
- 解析兜底：无 Action 但有实质文本 → 当作 Final Answer（防空 action 死循环）；
- 超 max_steps 强制收尾不抛错；工具失败转成 `[工具执行出错]` 观察文本回喂（自我修正）；
- 可观测性：graph_id / node_id 贯穿每条 trace。

### 3.3 技能注册表（skills.py）
```python
SKILL_REGISTRY = {
    "weather": {"aliases": ["天气"], "desc": "城市天气调研。任务参数=城市名",
                "tools": {"city_weather": (fn, desc)}, "prompt": WEATHER_PROMPT, "max_steps": 3},
    "file":    {"aliases": ["文档"], "desc": "文档加工（总结/翻译/提炼要点）",
                "tools": {"read_file": ..., "list_files": ...}, "prompt": FILE_PROMPT, "max_steps": 4},
}
```
- `resolve_skill()`：技能名/中文别名/前缀模糊匹配；
- 安全边界：file 技能所有路径操作限制在 `samples/` 内（`is_relative_to` 防目录穿越）。

### 3.4 并行派发引擎（dispatch.py）—— fan-out / fan-in
- `parse_spec()`：管道分隔 → 结构化契约列表；解析失败的分段返回 error 标记，
  由主 agent 看到错误后自行修正重试（不浪费 LLM 调用）；
- `dispatch_workers()`：构造 N 个 worker → **并行**（ThreadPoolExecutor，as_completed
  边完成边回收）/ **串行**（for 循环，A/B 基线）→ 契约回收 → 统计 → 汇总文本；
- 量化：`wall_clock`（并行墙钟）vs `serial_sum`（各 worker 时长之和 = 串行基线）；
- 防 context 爆炸：每个 worker 结果截短到 600 字喂回主 agent（完整 trace 仍在 shared_state）；
- 保护：单次派发最多 8 个 worker（防主 agent 一次派几十个）。

### 3.5 主编排 Agent（main_agent.py）
`MAIN_SYSTEM` 给出明确决策原则 + worked example：
- 任务可拆成 ≥2 个独立子任务（多城市/多文件/多对象）→ **必须** dispatch_workers；
- 单一子任务也走 dispatch_workers 派 1 个 worker（口径统一）；
- 拿不到技能先 list_skills 再决策。
主 agent 只做「拆解 → 派发 → 综合」——模型分层思想：路由决策与业务执行分离。

### 3.6 CLI 演示（demo.py）
输出：拓扑图（主节点 + worker 节点 + 各节点耗时 + 派发统计）、
节点 ReAct trace（`--trace 节点id`）、最终报告。串行模式 `--serial` 直接对照。

## 4. 实验结果（mock 模式实测，配 Key 后重跑即得真实 LLM 数字）

### 4.1 Parallel vs Serial A/B（3 题，真实工具耗时）
| 问题 | 并行墙钟 | 串行墙钟 | dispatch 加速 |
|------|---------|---------|--------------|
| 4 城天气对比（4 workers） | 7.46s | 22.54s | 3.58× |
| 3 份笔记总结（3 workers） | 3.02s | 6.65s | 3.00× |
| 5 城天气对比（5 workers） | 7.33s | 29.54s | 4.70× |
| **平均** | **5.94s** | **19.58s** | **3.76×** |

**结果解读**：
- worker 数越多、单 worker 耗时越均匀，加速越接近 N；
- 总墙钟加速 < 派发加速：主 agent 拆解/综合是串行段（Amdahl 定律），诚实教学点；
- worker 数由主 agent 自主决定（3/4/5 不等），非硬编码。

### 4.2 与 PPT 6.3/6.4 对应
- 拓扑：动态 Orchestrator-Workers（主 agent 派发，节点运行时生长）；
- 用图理由（6.4）：多异构节点协作（weather + file 两种 worker 同图）✓、
  可并行分支 ✓、需独立验证 ✓；
- 并行 vs 顺序：serial 基线正是 6.4「顺序任务」对照，量化并行收益；
- 落地要点：schema-first 交接 ✓、模型分层（主 agent 纯路由）✓、
  节点级可观测（graph_id/node_id）✓。

## 5. 优化方向

| 层面 | 方向 |
|------|------|
| 并行收益 | 主 agent 拆解/综合用更便宜模型或异步化，降串行段占比 |
| 技能扩展 | 注册表加 翻译/关键词提取/爬虫 等技能；技能带输入输出 schema 声明 |
| 决策稳定 | 主 agent 决策不稳时加规则兜底（问题含 N 个对象名 → 强制派发） |
| 工程 | worker 失败重试、结果去重、token 预算控制、trace 持久化回放 |
| 可视化 | 拓扑动画 / SSE 流式（week13 gateway.py 已有 SSE 基础可接） |

## 6. 关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| 主 agent 不派发 | 提示词没给 worked example | MAIN_SYSTEM 内置完整示例（问题→Thought→Action Input） |
| 空 action 死循环 | 模型拿到长结果直接写报告，不带 Final Answer 前缀 | _parse 兜底：无 Action 但有文本 → 当 Final Answer |
| Action Input 被污染 | 模型参数后补括号式旁白 | 跨行捕获后剔除末尾 `（...）` 旁白行 |
| dispatch 参数解析失败 | 模型幻觉技能名/格式 | parse_spec 返回 error 契约，Observation 回喂让模型自纠 |
| 并行墙钟不理想 | 有 worker 失败/超时 | 工具层全量 [ERROR] 降级不抛异常；超步数强制收尾 |
| mock 模式 worker 空转 | mock 大脑无限调 read_file | 按 history 里 Observation 个数分阶段（调工具→作答） |
| 包名遮蔽 | orchestrator/ 包与主模块同名导致 import 歧义 | 主模块更名 main_agent.py，演示脚本用 `from .main_agent import run` |
| 演示数据不入库 | .gitignore 忽略 data/ | 示例文档放 samples/（非忽略目录） |

## 7. 目录结构

```
week15/
├── orchestrator/
│   ├── llm_client.py       # 统一 LLM 客户端（llm_config + mock 模式）
│   ├── react_loop.py       # 通用 ReAct 引擎（主/worker 共用，可观测 trace）
│   ├── skills.py           # ★ 技能注册表（weather / file，worker 工厂）
│   ├── dispatch.py         # ★ 并行派发引擎（fan-out/fan-in + 契约）
│   ├── main_agent.py     # ★ 主编排 Agent（Supervisor）
│   ├── demo.py             # CLI 演示（拓扑图 + trace）
│   └── eval_compare.py     # parallel vs serial A/B
├── samples/                # 文档加工示例（notes_rag/agent/graph.md）
├── tests/test_orchestrator.py
├── outputs/eval_compare.json
├── README.md
└── ARCHITECTURE.md
```
