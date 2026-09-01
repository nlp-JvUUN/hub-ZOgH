# Week15 作业：可下发 Subagent 的编排 Agent —— 技能注册表驱动的并行派发系统

> 作业主题：自己实现一个可以下发 subagent 的 agent，并行完成多项工作。
> 本作业实现了一个 **Orchestrator-Workers 编排系统**（对应课件 Part 6 Graph Engineering）：
> 主 agent 自主把任务拆成多个子任务，按「技能名」派发并行 worker 执行，收齐后综合成最终答案。

---

## 一、先学习，再动手

**主 agent（ReAct + 派发工具）→ N 个并行 subagent（ReAct）→ fan-in 综合**。
**研究了老师课件**（`/python/week15graph和llm/week15 graph与LLM`，PPT Part 6 Graph Engineering）：
- L5 Graph Engineering = 设计多 agent 拓扑（Pipeline / Diamond / Orchestrator-Workers 三种典型拓扑）；
- 何时用图：任务可分、多节点协作、可并行分支、需独立验证；
- 落地要点：**schema-first 交接**（边上有结构化数据契约，下游不靠猜）、**模型分层**
  （便宜模型做路由，贵模型做合成）、**节点级可观测**（graph_id/run_id/node_id 审计）；
- Orchestrator-Workers：主管派发 + 回收合成（Anthropic Research 采用）。

**延续自己的 week13 脉络**：week13 的 SkillFlow 是「单 agent 调技能」的 harness（L4 Loop），
本作业让它长出**派发能力**（L5 Graph）：主 agent 把「调一个技能」升级为「派发多个 worker
并行跑任务」——同一套技能清单思想，从单循环走向多 agent 拓扑。

---

## 二、思路：把「并行编排」做成可测量的系统

```
用户问题
   ↓
主编排 Agent（ReAct，只有 list_skills + dispatch_workers 两个工具）
   ├─ 简单任务 → dispatch_workers 派 1 个 worker（口径统一）
   └─ 可拆分任务 → dispatch_workers("weather: 北京 | weather: 上海 | file: 总结...")
                      ↓ fan-out（ThreadPoolExecutor 并行）
              ┌─ worker1 ReAct（weather 技能：city_weather 工具）──┐
              ├─ worker2 ReAct（weather 技能：city_weather 工具）──┤ 并行
              └─ worker3 ReAct（file 技能：read_file/list_files）──┘
                      ↓ fan-in（结构化契约回收，截短防 context 爆炸）
             主编排 Agent 综合成最终报告（含并行加速统计）
```

**核心价值**：
并行的意义不是少做事，而是把 N 个独立子任务的墙钟从 **sum** 压到 **≈max**。

## 三、交付内容

```
week15/
├── README.md                        # 本文件：思路 + 差异 + 复现
├── ARCHITECTURE.md                  # ★ 详细架构（范式归属/设计决策/踩坑）
├── orchestrator/                    # ★ 编排系统核心（纯标准库）
│   ├── llm_client.py                #   统一 LLM 客户端（复用 llm_config.py + mock 模式）
│   ├── react_loop.py                #   通用 ReAct 引擎（主/worker 共用，graph_id/node_id 可观测）
│   ├── skills.py                    #   ★ 技能注册表：weather + file 双技能（worker 工厂）
│   ├── dispatch.py                  #   ★ 并行派发引擎：fan-out/fan-in + schema-first 契约
│   ├── main_agent.py              #   ★ 主编排 Agent（Orchestrator-Workers 的 Supervisor）
│   ├── demo.py                      #   CLI 演示：拓扑图 + 节点 trace + 并行统计
│   └── eval_compare.py              #   parallel vs serial A/B 量化对比
├── samples/                         # 文档加工场景的示例笔记（3 份，可复现）
├── tests/test_orchestrator.py       # 15 个单测（离线可跑，不碰 LLM/网络）
└── outputs/eval_compare.json        # A/B 实测原始数据（本地生成，不入库）
```

## 四、快速体验

```bash
cd week15

# 1. 单元测试（离线，无需 API Key）
python3 -m unittest tests.test_orchestrator -v

# 2. 命令行演示（未配 Key 时自动进入 mock 模式：脚本化大脑 + 真实工具）
python3 -m orchestrator.demo                          # 天气对比（默认）
python3 -m orchestrator.demo --question-file          # 文档加工场景
python3 -m orchestrator.demo "你的问题" --serial      # 串行模式（A/B 基线）

# 3. 真实 LLM 模式（推荐）：在 曾文静/.env 填入 DEEPSEEK_API_KEY 后自动切换
python3 -m orchestrator.eval_compare                  # 3 题 × 并行/串行 A/B
```

**API Key 说明**：`llm_client.py` 复用仓库根的 `llm_config.py`（读取 `曾文静/.env`）。
未配置 Key 时自动进入 **mock 模式**（脚本化 LLM + 真实工具：天气 API 真实调用、
文件真实读取），全流程可离线跑通并演示并行收益；配置 Key 后自动切换真实模型。

## 五、实验结果（当前为 mock 模式实测，配 Key 后重跑即得真实 LLM 数字）

`python3 -m orchestrator.eval_compare` 输出（3 题 × 并行/串行）：

| 问题 | 并行墙钟 | 串行墙钟 | 派发加速 |
|------|---------|---------|---------|
| 4 城天气对比（4 workers） | 7.46s | 22.54s | 3.58× |
| 3 份笔记总结（3 workers） | 3.02s | 6.65s | 3.00× |
| 5 城天气对比（5 workers） | 7.33s | 29.54s | 4.70× |
| **平均** | **5.94s** | **19.58s** | **3.76×** |

**结果解读**：
- 派发加速 **3.76×**：N 个独立 worker 并行，墙钟从 sum 压到 ≈max；
- 加速比随 worker 数上升（3.58× → 4.70×），因为并行段占比更高；
- 总墙钟加速小于派发加速：主 agent 自己的拆解/综合是串行段，不参与并行
  （Amdahl 定律——可并行部分才受益，这是诚实且重要的教学点）。

## 六、自测结果

```
python3 -m unittest tests.test_orchestrator -v
Ran 15 tests in 1.8s  →  OK
```
覆盖：派发参数解析（别名/模糊匹配/错误兜底）、技能注册表完整性、路径穿越拦截、
ReAct 解析（跨行 Action Input/Final Answer/兜底）、并行 vs 串行墙钟差异（假 worker 验证加速逻辑）。
