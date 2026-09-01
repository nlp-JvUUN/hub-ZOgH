# Week14 作业：Skill 的自进化优化 —— token 消耗 × 执行效率 × 无损质量

## 一、思路：把"自进化"做成测量闭环

```
Round 0  大模型编写 skill v1（纯 LLM 方案，一次调用全量处理）
   ↓ benchmark 测量：2814 tokens/次
Round 1  第 1 轮优化 v2（脚本做机械工作 + SKILL 精简）
   ↓ benchmark 测量：2602 tokens/次，但【近似重复漏检 2 对】——质量回退！
Round 2  第 2 轮自进化 v3（精简 plan + 模糊去重 + 决策式组装 + 无损门禁）
   ↓ benchmark 测量：1141 tokens/次（-59.4%），--verify ✅ 无损
   ✓ 收敛：质量与 v1 持平，成本降到 40%
```

关键转折：**v2 表面省了 token，实则质量回退**（2 对近似重复未合并）。
如果只看 token 数字就会误以为优化成功 —— 所以每轮都必须同时测质量。
这正是"自进化 Agent"与"普通 prompt 调优"的区别：**基于证据迭代，且质量可验证**。

---

## 二、交付内容

```
week14/
├── README.md                        # 本文件：思路 + 差异 + 复现
├── COMPARISON.md                    # ★ 优化前后对比报告（数字 + 图表 + 结论）
├── skills/notes-consolidator/       # ★ 被优化的 skill（自进化产物）
│   ├── SKILL.md                     #   最终版（v3，当前生效）
│   ├── SKILL_v1.md                  #   初版存档（大模型原创，冗长版）
│   ├── SKILL_v2.md                  #   第 1 轮优化存档
│   ├── EVOLUTION.md                 # ★ 自进化记录：每轮改动 + 测量反馈 + 决策
│   ├── scripts/consolidate.py       #   v3 脚本：精简 plan / 模糊去重 / 组装 / --verify
│   ├── scripts/consolidate_v2.py    #   v2 脚本存档：切章节 / 完全重复检测
│   └── data/                        #   示例语料（含完全重复与近似重复，可复现）
│       ├── raw_notes.md             #     小语料（934 tokens，主基准）
│       └── raw_notes_large.md       #     大语料（2339 tokens，扩展性验证）
├── bench/
│   ├── benchmark.py                 # ★ 测量 harness：token/耗时/质量三合一
│   ├── scaling.py                   # ★ 多语料扩展性验证（借鉴洪建宇的多场景思路）
│   ├── results.json                 #   测量原始数据
│   └── tmp/                         #   plan 中间产物（复现用）
└── outputs/
    ├── v1.md                        # v1 模型输出（全量整理稿）
    ├── v2.md                        # v2 模型输出（含 2 对重复 ❌）
    ├── v3.md                        # v3 组装输出（小语料，无损 ✅）
    ├── v3_large.md                  # v3 组装输出（大语料，无损 ✅）
    ├── decisions_v3.json            # v3 中 LLM 唯一要写的"决策"（小语料）
    └── decisions_large.json         # v3 决策（大语料）
```

## 三、如何复现

```bash
# 1. 生成两份 plan 并测量（含脚本耗时、token、质量断言）
PYTHONPATH=<tiktoken安装路径> python3 bench/benchmark.py

# 2. 多语料扩展性验证（验证 plan 方案随语料增大优势更明显）
PYTHONPATH=<tiktoken安装路径> python3 bench/scaling.py

# 3. 单独跑 v3 全流程（plan → 组装 → 无损校验）
python3 skills/notes-consolidator/scripts/consolidate.py \
    skills/notes-consolidator/data/raw_notes.md bench/tmp/plan_v3.json
python3 skills/notes-consolidator/scripts/consolidate.py \
    skills/notes-consolidator/data/raw_notes.md bench/tmp/plan_v3.json \
    --assemble outputs/decisions_v3.json -o outputs/v3.md
python3 skills/notes-consolidator/scripts/consolidate.py \
    skills/notes-consolidator/data/raw_notes.md bench/tmp/plan_v3.json \
    --verify outputs/v3.md      # → ✅ 无损
```

## 四、扩展性验证

| 语料 | 正文 tokens | plan tokens | plan/正文 | v1 单次成本 | v3 单次成本 | 节省 |
|------|------------:|------------:|----------:|------------:|------------:|-----:|
| raw_notes.md（小） | 934 | 489 | 0.52 | 2828 | 1141 | **59.7%** |
| raw_notes_large.md（大） | 2339 | 769 | 0.33 | 5287 | 1484 | **71.9%** |

原因：v1 的成本与**正文规模**线性相关（全文进上下文）；v3 的 plan 只与**需要判断的量**
（章节数、重复数）相关。语料越大，省得越多。他的双模式测量（无 key 估算 / 有 key 真实
usage）也值得借鉴：本作业用 tiktoken/cl100k_base（真实 BPE 口径，比"中文×1.5"的粗估算准），
client.py 加一个可选 live 模式。

## 六、核心结论

1. **token 消耗 -59.4%**：2814 → 1141 tokens/次（SKILL -58%、输入 -48%、输出 -77%），
   且全部为实测（tiktoken/cl100k_base）。
2. **执行效率提升**：LLM 推理估算 -59%，本地脚本仅 ~72ms；LLM 从"抄写员"变成"法官"。
3. **质量无损且有门禁**：0 内容丢失、0 重复残留、标题全覆盖，`--verify` 自动化保证。
4. **方法论收获**：优化必须"测量 + 质量门禁"双轨并行 —— v2 的教训证明，
   只看 token 数字会掩盖质量回退；自进化的价值在于每轮基于证据改进。
