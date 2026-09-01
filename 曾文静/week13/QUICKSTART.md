# SkillFlow 快速上手

零第三方依赖，Python 3.9+。所有命令在 `week13/` 目录下执行。

## 1. 渐进式加载

```bash
# 增量扫描：首次解析全部，第二次只报告变化（缓存命中）
python -m skillflow scan
python -m skillflow scan          # 变化 0 个

# 单技能详情：注意「实现已加载 = False」—— 发现 ≠ 加载
python -m skillflow info word-count

# 加载预算：budget=3 < fetch-source 的 weight=5 → 推迟执行
python -m skillflow --budget 3 run fetch-source
# 💤 [fetch-source] deferred: 加载预算不足：fetch-source 需要 weight=5，剩余预算 3

# 热更新：watch 模式运行中，向 skills/ 放入新目录即可被自动发现
python -m skillflow watch
# 另一个终端：mkdir skills/my-new && 写入 SKILL.md + skill.py
# watch 终端会打印 "热更新: my-new ..."，无需重启
```

## 2. 渐进式执行

```bash
# 生成器技能：进度一条条流出来
python -m skillflow run slow-progress steps=5
#   ▸ [slow-progress] 20% (1/5) 任务 第 1/5 步完成
#   ▸ [slow-progress] 40% (2/5) 任务 第 2/5 步完成
#   ...

# 管道：数据按 consumes/provides 契约逐级流动
python -m skillflow pipe "fetch-source | word-count | format-report"
# fetch-source 读 L3 资源 → text 注入 word-count → count 注入 format-report

# 三种失败策略对比
python -m skillflow pipe "flaky-demo | word-count" should_fail=true --policy stop     # 中止
python -m skillflow pipe "flaky-demo | word-count" should_fail=true --policy skip     # 级联跳过
python -m skillflow pipe "flaky-demo | format-report" should_fail=true --policy default
```

## 3. 心跳与记忆

```bash
# 立即触发一次全部心跳技能（daily-report 执行 Memory Flush）
python -m skillflow heartbeat --once

# 查看记忆
python -m skillflow memory      # MEMORY.md（纪要）
python -m skillflow journal     # 今日日志（录音）

# 手动触发 Memory Flush
python -m skillflow flush
```

## 4. HTTP 网关（SSE 实时事件流）

```bash
python -m skillflow serve --port 8620
```

```bash
# 健康检查 / 技能清单（只含 L1 元数据）
curl http://127.0.0.1:8620/api/health
curl http://127.0.0.1:8620/api/skills

# 创建会话并投递一条管道消息（同一会话内严格串行）
SID=$(curl -s -X POST http://127.0.0.1:8620/api/sessions \
      -d '{}' -H 'Content-Type: application/json' \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

curl -s -X POST http://127.0.0.1:8620/api/sessions/$SID/messages \
     -d '{"pipe":"slow-progress | word-count","inputs":{"steps":4,"text":"hello world"}}' \
     -H 'Content-Type: application/json'

# 实时事件流（SSE）：过程可见，结果未出先见进度
curl -N http://127.0.0.1:8620/api/sessions/$SID/stream?after=0

# 增量轮询（普通 HTTP）
curl "http://127.0.0.1:8620/api/sessions/$SID/events?after=0"

# 热更新 / 心跳 / Memory Flush 的 HTTP 入口
curl -X POST http://127.0.0.1:8620/api/reload
curl -X POST http://127.0.0.1:8620/api/heartbeat/run
curl -X POST http://127.0.0.1:8620/api/flush
curl http://127.0.0.1:8620/api/memory
```

## 5. ReAct 元技能（LLM 自然语言调度，可选）

```bash
# 复用根目录 llm_config.py：在 曾文静/.env 里配好 Key 即可（模板 .env.example）
#   DEEPSEEK_API_KEY=sk-xxx
# 换模型：环境变量 LLM_MODEL / DEEPSEEK_MODEL，或改 llm_config.DEFAULT_MODEL

# 自然语言入口：agent-react 技能选择并调用其他技能，多轮推理后给出回答
python -m skillflow chat "统计这段话的单词数：hello world skillflow"
python -m skillflow chat "帮我生成一段 fetch-source 管道的统计报告"

# REPL 里同样可用
python -m skillflow repl
# sf> chat 统计一下 sample.txt 的字数
```

未配置 Key 时给出明确报错；工具执行失败会作为「观察」回喂模型恢复。
（LLM 配置统一走根目录 `llm_config.py`，skill 内不重复配置。）

## 6. 卸载已加载实现（释放内存）

```bash
# CLI 每次是新进程，卸载主要在长驻进程里体现价值：REPL / serve
python -m skillflow repl
# sf> run slow-progress steps=1
# sf> loaded            # 已加载实现: ['slow-progress']
# sf> unload slow-progress
# sf> loaded            # 已加载实现: （无）
```

## 7. REPL

```bash
python -m skillflow repl
# sf> skills
# sf> run word-count text="hello"
# sf> pipe fetch-source | word-count | format-report
# sf> chat 统计一下 sample.txt 的字数
# sf> budget 3          # 动态调小加载预算
# sf> loaded / unload   # 查看 / 卸载已加载实现（L2）
# sf> flush / journal / memory / sessions
# sf> quit
```

## 8. 自测

```bash
python -m unittest discover -s tests -v     # 27 个用例，覆盖四条主轴
```

## 演示脚本（一次性跑完）

```bash
python -m skillflow scan
python -m skillflow run slow-progress steps=3
python -m skillflow pipe "fetch-source | word-count | format-report"
python -m skillflow --budget 3 run fetch-source
python -m skillflow pipe "flaky-demo | word-count" should_fail=true --policy stop
python -m skillflow heartbeat --once
python -m skillflow memory
# （可选，需在根目录 .env 配置 DEEPSEEK_API_KEY，见 llm_config.py）
python -m skillflow chat "统计这句话的单词数：hello world"
```
