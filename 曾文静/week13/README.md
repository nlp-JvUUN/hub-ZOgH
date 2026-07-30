# week13 — SkillFlow：渐进式 Skills 加载执行 Harness

作业主题：**写一套可以实现「渐进式加载执行 skills」的 harness**。

SkillFlow 把「渐进式」拆成三条可验证的主轴

| 主轴 | 含义 | 落点 |
|------|------|------|
| **加载渐进** | 发现 → 元数据 → 实现 → 资源，按需推进，只加载变化的部分 | `discovery.py` + `loader.py` |
| **执行渐进** | 生成器流式产出，结果未出、过程可见；管道逐级对接、失败降级 | `engine.py` |
| **调度渐进** | ReAct 循环本身是一个技能（agent-react）：LLM 用 L1 元数据选技能、填充参数、观察结果、多轮推理 | `skills/agent-react/` |
| **交付渐进** | Lane 串行会话、HEARTBEAT 主动调度、Markdown 记忆日志 + Memory Flush | `session.py` + `scheduler.py` + `journal.py` |

设计约束（刻意与参考作业拉开差异）：

- **零第三方依赖**：纯 Python 标准库（参考作业依赖 sqlite3 + yaml + asyncio 全家桶）；
- **同步生成器流**，不用 asyncio —— 渐进式执行靠「生成器 yield」，不靠事件循环；
- **契约式数据对接**：管道按 SKILL.md 里声明的 `consumes` / `provides` 对接，
  而不是参考作业「skill 名转下划线当参数名」的命名约定；
- **执行记录落 Markdown**（人机双读，可直接 git diff），而不是 SQLite；
- **L1 与 L2 彻底分离**：发现（registry）永不 import 任何 skill 实现，
  参考作业在初始化阶段就把全部实现类 import 进内存。

---

## 目录结构

```
week13/
├── skillflow/                  # harness 核心包（零第三方依赖）
│   ├── model.py                # 数据模型：SkillSpec / Progress / Event / ExecutionReport
│   ├── discovery.py            # L0/L1：增量 manifest 扫描 + frontmatter 解析 + 热更新
│   ├── loader.py               # L2/L3：实现与资源懒加载 + LoadBudget 加载预算
│   ├── engine.py               # 渐进式执行引擎：生成器流 + 管道 + 失败策略
│   ├── session.py              # 会话层：Lane 串行队列 + InternalMessage（Fat Gateway 模式）
│   ├── journal.py              # Markdown 每日日志 + JSONL + Memory Flush
│   ├── scheduler.py            # HEARTBEAT 心跳调度器
│   ├── app.py                  # HarnessApp：把以上组件组装成一台完整 harness
│   ├── gateway.py              # HTTP 网关（含 SSE 实时事件流）
│   └── cli.py / __main__.py    # CLI / REPL 入口
├── skills/                     # 演示 skills（SKILL.md + skill.py）
│   ├── agent-react/            # ReAct 元技能：LLM 自然语言调度（LLM 复用根目录 llm_config.py）
│   ├── fetch-source/           # weight=5，读 L3 资源，演示加载预算与数据入口
│   ├── word-count/             # 词频统计（普通函数式技能）
│   ├── format-report/          # 报告排版（管道终端技能）
│   ├── slow-progress/          # 生成器技能，逐步 yield 进度（渐进式输出）
│   ├── flaky-demo/             # 失败注入，演示 stop/skip/default 三种策略
│   └── daily-report/           # heartbeat: 30s，触发 Memory Flush（HEARTBEAT.md 概念）
├── tests/test_harness.py       # 27 个自测用例（python -m unittest）
├── README.md / ARCHITECTURE.md / QUICKSTART.md
└── state/  journal/            # 运行时生成（.gitignore 已排除）
```

## 快速体验

```bash
cd week13
python -m skillflow scan                    # 增量扫描（第二次 0 变化）
python -m skillflow run slow-progress steps=3          # 渐进式输出
python -m skillflow pipe "fetch-source | word-count | format-report"   # 管道
python -m skillflow --budget 3 run fetch-source        # 预算不足 -> deferred
python -m skillflow pipe "flaky-demo | word-count" should_fail=true --policy stop
python -m skillflow heartbeat --once        # 心跳技能 -> Memory Flush
python -m skillflow serve --port 8620       # HTTP 网关（SSE 实时事件流）
python -m skillflow repl                    # 交互式命令行
python -m skillflow chat "统计这段文字的单词数：hello world"   # ReAct 元技能（需根目录 .env 配置 Key）
python -m unittest discover -s tests -v     # 自测
```

