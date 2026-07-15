# Week 11 — 天气查询工具调用的"循环化"改造（Agent Loop）

## 一、作业思路：先看懂"单轮"，再理解"循环"

课件 `mode_function_call/run_function_call.py` 的闭环是**单轮**的：

```
User → LLM（输出 tool_call）→ 宿主执行工具 → 结果回填 → LLM 再答一次 → 结束
```

它的问题（课件 ARCHITECTURE.md 也点明了）：

1. **只能调一轮工具**：模型一次最多"规划一步"。比如"宁德时代的营收 + 总部天气"这种问题，模型想"先查营收，再查天气，若营收段落不全还想再查一次"——**做不到**，单轮无法再调；
2. **工具报错即对话中断**：工具返回错误时，模型没有机会修正重试；
3. **工具粒度决定能力上限**：`get_weather(city)` 把"城市名→经纬度"和"经纬度→天气"揉在一个函数里，模型无法只问坐标、无法复用坐标再查别的（如空气质量）。

**循环调用（Agent Loop）** 就是把第 2 步到第 4 步包进一个循环，直到模型自己决定停：

```
User → [ LLM → tool_call → 执行 → 回填 ] × N → LLM 最终回答
          └────── 每一轮都由模型自主决定 ──────┘
```

循环让 LLM 获得"多步推理 + 试错修正"的能力，这正是 Agent（而不是单次函数调用）的核心。

---

## 二、本作业的设计

### 1. 工具拆分：`get_weather(city)` → 5 个原子工具

| 工具 | 输入 → 输出 | 单独使用场景 | 链式场景 |
|------|------------|-------------|---------|
| `geocode` | 城市名 → 结构化位置 JSON（含经纬度） | 问"某城市的经纬度" | 任何城市类问题的第一步 |
| `get_current_weather` | 经纬度 → 当前天气 | 用户直接给坐标 | geocode 之后 |
| `get_daily_forecast` | 经纬度 + days → 未来 N 天预报 | 用户直接给坐标问预报 | geocode 之后 |
| `get_air_quality` | 经纬度 → AQI/PM2.5/PM10 等 | 用户直接给坐标问空气 | geocode 之后 |
| `get_comfort_index` | 温度/湿度/风速 → 体感舒适度 | ——（**下游衍生工具**） | geocode → 天气 → **提取参数** → 舒适度 |

拆分粒度比常见的"2 工具"方案更细：**只问今天**不用拉逐日预报、**只问空气**不用拉天气，且**天气+空气可以同一轮并行调两个工具**。工具输出是结构化 JSON，可直接作为下一个工具的输入（geocode 的 JSON 里就有 latitude/longitude）。

### 2. 循环状态机（`AgentLoop.run`）

```
for step in 1..MAX_STEPS:
    decision = LLM(messages)                # 决策：调工具 or 给最终答案
    if decision 无 tool_calls:  break       # ① 模型主动终止（正常出口）
    messages += assistant(tool_calls)       # 回填模型消息（含 tool_call_id）
    for c in decision.tool_calls:           # 一轮可并行多个工具
        result = TOOL_DISPATCH[c.name](**c.args)
        messages += tool(c.id, result)      # 结果回填，继续下一轮
```

### 3. 三层终止保护（循环不能"死循环"）

| 保护 | 机制 |
|------|------|
| ① 模型主动停 | 某轮不再输出 `tool_calls`，该轮文本即最终答案 |
| ② 最大轮数 | `--max-steps`（默认 8），超限强制终止并说明 |
| ③ 死循环检测 | 连续多轮调用**同一工具+同一参数**视为死循环，强制终止（防止模型反复重试同一个错误参数烧 token） |

### 4. 失败自愈：循环相对单轮的关键价值

工具层（`weather_tools.py`）**不抛异常**，所有失败统一返回以 `[ERROR]` 开头的文本（含错误码 + 修正建议）。这些错误会作为 tool 结果回填给模型，模型在**下一轮**自主决定怎么处理：

- `[ERROR][NOT_FOUND]`（城市不存在）→ 修正写法重试，或如实告知用户（不编造坐标）；
- `[ERROR][PARAM]`（坐标越界/非数字）→ 修正参数重试；
- `[ERROR][NETWORK]`（网络失败）→ 如实说明。

单轮闭环遇到同样的错误，只能把错误原文丢给用户；循环里模型可以"失败 → 反思 → 重试 → 成功"。

### 5. 可观测性

- 逐轮打印：`[tool round N] 工具名(参数)` + 结果摘要 + `[llm round N]`；
- `--transcript file.jsonl` 导出完整循环轨迹（每轮的 tool_calls、结果、token 统计），可复现、可检查；
- 每次运行汇总：工具调用次数、循环轮数、token 消耗、总耗时。

### 6. 双驱动设计：`--mock` 模拟决策器

`MockPlanner` 用规则脚本模拟 LLM 的决策，输出结构与真实模型一致，因此 `AgentLoop` 不需要区分驱动：

- **没有 API Key 也能完整跑通循环机制**（链式 / 下游衍生提取 / 并行 / 自愈 / 拒答六个场景都有脚本覆盖）；
- 先用 mock 离线验证循环逻辑，再切真实模型，问题定位更清晰。



## 三、快速开始

### 依赖与环境变量

```bash
pip install openai httpx

# Linux / macOS
export DEEPSEEK_API_KEY="sk-xxx"      # 真实模型模式需要
# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="sk-xxx"
```

### 运行

```bash
# ① 工具层自测（真实 Open-Meteo API，无需 Key）
python weather_tools.py

# ② 模拟驱动演示循环机制（无需 API Key，推荐先跑这个）
python agent_loop.py --demo --mock

# ③ 真实模型单问题
python agent_loop.py -q "宁德今天的天气怎么样？"
python agent_loop.py -q "经度119.52、纬度26.66 的当前天气和空气质量？"

# ④ 真实模型内置示例集（覆盖链式/预报/空气/体感舒适度/并行/拼写自愈/拒答）
python agent_loop.py --demo

# ⑤ 导出循环轨迹
python agent_loop.py -q "北惊的天气怎么样？" --transcript transcript.jsonl

# ⑥ 切换模型提供商（与课件一致）
python agent_loop.py -q "北京的天气" --provider dashscope
```

### 模拟驱动实测输出（本机 2026-08-31 实跑摘录）

```
Q3：宁德今天体感舒适吗？
  → [tool round 1] geocode({'city': '宁德'})
    ↩ {"name": "宁德市", ..., "latitude": 26.66167, "longitude": 119.52278}
  → [tool round 2] get_current_weather({'latitude': 26.66167, 'longitude': 119.52278})
    ↩ 当前天气：小毛毛雨 温度：29.6°C 相对湿度：79% 风速：18.7 km/h
      ↻ [mock 提取] 从天气文本提取 温度=29.6 湿度=79 风速=18.7
  → [tool round 3] get_comfort_index({'temperature': 29.6, 'humidity': 79.0, 'wind_speed': 18.7})
    ↩ 体感温度约 30.8°C（舒适度等级：炎热）...
  → [llm round 4] 不再调用工具，输出最终回答
（model_stop：工具调用 3 次，循环 4 轮，耗时 3.0s）
```

（`--demo --mock` 共 6 个场景：链式 ×3、下游衍生提取、并行、失败自愈、诚实拒答，全部通过。）

---

## 四、文件结构

```
week11/
├── README.md            # 本文件：作业思路 + 差异说明 + 使用文档
├── weather_tools.py     # 工具层：4 个原子工具（纯业务，LLM 无关，可独立测试）
└── agent_loop.py        # Agent Loop 主程序：循环状态机 + 三层终止保护
                         #   + 失败自愈 + 轨迹导出 + --mock 模拟驱动
```

## 五、教学要点小结

1. **工具原子化**：把"城市名→天气"一步拆成"定位"与"查询"两个阶段，模型才能组合出更多玩法（坐标复用、并行查询）；
2. **LLM 主导**：调哪个工具、调几个、调几次、何时停，全部由模型在循环里自主决定，宿主只做执行与回填；
3. **循环的三大价值**：多步链式推理、失败自愈、按需并行——单轮闭环全都做不到；
4. **循环必须有兜底**：最大轮数 + 死循环检测，防止模型无限循环烧 token；
5. **协议层与业务层分离**：`weather_tools.py` 不依赖任何 LLM 库，可独立测试、独立部署、随时换 Agent 框架。
