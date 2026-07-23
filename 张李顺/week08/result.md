# vLLM Demo 结果汇总

文件名：`vllm_demo_results_20260707_summary.md`

模型：`qwen2-0.5b`

本文件汇总本轮 vLLM 演示的最终结果。内容只保留每份 demo 的关键指标和简单分析。

---

## 1. guided_choice：枚举约束

### 最终结果

| 指标 | 裸 prompt | guided_choice |
|---|---:|---:|
| 输出合法（在枚举内） | 10/12（83%） | 12/12（100%） |
| 预测正确 | 3/12（25%） | 3/12（25%） |
| 平均延迟 | 0.071 秒 | 0.298 秒 |

### 简单分析

guided_choice 保证输出一定落在候选标签内，所以输出合法率达到 100%。本次模型明显偏向输出“查股价”，所以 guided_choice 没有提升分类准确率。

---

## 2. guided_regex：正则约束

### 任务 1：日期标准化

目标格式：`YYYY-MM-DD`

| 指标 | 裸 prompt | guided_regex |
|---|---:|---:|
| 格式合法率 | 5/6（83%） | 6/6（100%） |

### 任务 2：A 股代码抽取

目标格式：6 位数字，正则：`\d{6}`

| 指标 | 裸 prompt | guided_regex |
|---|---:|---:|
| 格式合法率 | 2/5（40%） | 5/5（100%） |

### 简单分析

guided_regex 能保证输出满足固定格式，适合日期、股票代码、手机号、邮编等字段。它保证格式合法，但不保证语义一定正确，例如 `000001` 被输出成 `600001` 仍然属于格式合法但内容错误。

---

## 3. guided_json：JSON Schema 约束

任务：财报问答意图抽取，输出 `{company, year, metric}`。

### 最终结果

| 指标 | 裸 prompt | response_format | guided_json |
|---|---:|---:|---:|
| 合法 JSON | 9/9（100%） | 9/9（100%） | 9/9（100%） |
| 字段齐全 | 9/9（100%） | 9/9（100%） | 9/9（100%） |
| year 在 2015~2025 | 8/9（89%） | 8/9（89%） | 9/9（100%） |
| metric 在枚举内 | 9/9（100%） | 9/9（100%） | 9/9（100%） |
| jsonschema 完全通过 | 8/9（89%） | 8/9（89%） | 9/9（100%） |

### 简单分析

response_format 能让输出是 JSON，但不能强制字段满足业务 schema。guided_json 能约束字段、类型、范围、枚举，所以本次完整 schema 通过率达到 100%。

---

## 4. response_format：OpenAI 标准 JSON 模式

任务：新闻情绪与关键词抽取。

### 最终结果

| 指标 | 裸 prompt | response_format |
|---|---:|---:|
| 合法 JSON | 5/5（100%） | 5/5（100%） |
| 有 sentiment 字段 | 5/5（100%） | 5/5（100%） |
| sentiment 值合法 | 5/5（100%） | 5/5（100%） |
| 有 confidence 字段 | 5/5（100%） | 5/5（100%） |
| 有 keywords 字段 | 5/5（100%） | 5/5（100%） |

### 简单分析

本批样本里 raw 和 response_format 都表现很好，所以差距没有拉开。response_format 的核心价值是保证 JSON 对象输出；如果需要字段类型、枚举、范围都严格正确，应使用 guided_json。

---

## 5. Function Call：工具参数生成

任务：模拟 Function Call 的 arguments 生成，对比裸 prompt、response_format、guided_json。

### 工具 1：get_stock_quote

| 指标 | 裸 prompt | response_format | guided_json |
|---|---:|---:|---:|
| JSON 语法合法 | 43/50（86%） | 50/50（100%） | 50/50（100%） |
| 必选字段齐全 | 43/50（86%） | 50/50（100%） | 50/50（100%） |
| 完整 schema 通过 | 30/50（60%） | 34/50（68%） | 50/50（100%） |
| 平均延迟 | 0.701 秒 | 0.683 秒 | 0.734 秒 |

### 工具 2：create_order

| 指标 | 裸 prompt | response_format | guided_json |
|---|---:|---:|---:|
| JSON 语法合法 | 48/50（96%） | 50/50（100%） | 50/50（100%） |
| 必选字段齐全 | 48/50（96%） | 50/50（100%） | 50/50（100%） |
| 完整 schema 通过 | 21/50（42%） | 21/50（42%） | 50/50（100%） |
| 平均延迟 | 0.953 秒 | 0.971 秒 | 1.023 秒 |

### 简单分析

Function Call 的关键是生成可执行的参数 JSON。response_format 提高了 JSON 语法合法率，但字段值仍可能违反枚举、正则、非空数组等约束；guided_json 让两个工具的完整 schema 通过率都达到 100%。

---

## 6. Throughput Benchmark：吞吐性能对比

任务：50 个 prompt，每个最多生成 100 个新 token。

### 最终结果

| 模式 | 总耗时 | QPS | tokens/s | 相对 vLLM |
|---|---:|---:|---:|---:|
| transformers 串行 | 105.01 秒 | 0.48 | 35 | 0.02× |
| transformers batch=8 | 21.17 秒 | 2.36 | 175 | 0.11× |
| vLLM 批处理 | 2.23 秒 | 22.47 | 1569 | 1.00× |

### 加速比

| 对比项 | 加速 |
|---|---:|
| vLLM 相对 transformers 串行 | 47.2× |
| vLLM 相对 transformers batch=8 | 9.5× |

### 简单分析

transformers batch 已经明显快于串行，因为它一次处理多条样本。vLLM 进一步依靠 PagedAttention 和 continuous batching 提高吞吐，本次 tokens/s 从 175 提升到 1569。

---

## 总体结论

| 技术 | 主要解决的问题 | 本次结果 |
|---|---|---|
| guided_choice | 输出必须属于候选枚举 | 合法率 100%，但准确率仍取决于模型理解 |
| guided_regex | 输出必须匹配固定格式 | 日期和股票代码格式合法率均达到 100% |
| response_format | 输出必须是 JSON 对象 | JSON 合法性强，但不保证完整业务 schema |
| guided_json | 输出必须符合 JSON Schema | 多个结构化任务 schema 通过率达到 100% |
| Function Call | 工具参数必须可解析、可执行 | guided_json 对 arguments 生成最稳定 |
| vLLM 批处理 | 推理吞吐提升 | 相比 transformers batch=8 加速 9.5× |

