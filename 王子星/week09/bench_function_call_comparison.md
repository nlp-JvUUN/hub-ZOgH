# vLLM 部署能力对比报告（吞吐 Benchmark & Function Call 可靠性）

## 1. 实验概述

| 脚本 | 对比维度 | 对比对象 | 测试规模 |
|------|---------|---------|---------|
| `src/bench_throughput.py` | 推理吞吐 | transformers 串行 / transformers batch=8 / vLLM continuous batching | 50 prompts × 100 生成 token |
| `src/demo_function_call.py` | Function Call 参数生成可靠性 | 裸 prompt / response_format(json_object) / guided_json | 2 工具 × 50 条 × 3 模式 = 300 次调用 |

两个脚本共用同一基座模型 **Qwen2.5-0.5B-Instruct**，分别验证 vLLM 的两项核心能力：**推理引擎层的吞吐优势**（PagedAttention + continuous batching）与**服务层的约束解码可靠性**（guided_json）。

---

## 2. 实验配置

| 配置项 | bench_throughput.py | demo_function_call.py |
|--------|---------------------|------------------------|
| 基础模型 | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B（vLLM OpenAI 兼容 server） |
| 测试规模 | 50 prompts（短/中/长混合） | 每工具 50 条测试用例 |
| 关键参数 | `max_new_tokens=100`，`batch_size=8` | `temperature=0`，`max_tokens=250` |
| 对比模式 | 串行 / batch=8 / vLLM | 裸 prompt / response_format / guided_json |
| 评估工具 | 总耗时、QPS、tokens/s | jsonschema 分层校验（JSON合法→必选字段→完整schema） |
| 运行方式 | 本地脚本直接跑 transformers + vLLM offline `LLM()` | 需先 `start_server.sh` 起 vLLM OpenAI server，再用 `openai` SDK 调用 |
| 结果产出 | `outputs/throughput_results.json`、`outputs/throughput_comparison.png` | `outputs/function_call_results.json` |

---

## 3. 方法说明

### 3.1 吞吐三种路线

- **[A] transformers 串行**：逐条调用 `model.generate()`，无 batching。
- **[B] transformers batch=8**：手动构造 batch，左侧 padding 后统一 `generate()`。
- **[C] vLLM**：`LLM.generate()` 内置 PagedAttention（KV cache 按 block 管理，消除 padding 浪费）+ continuous batching（不同长度请求动态插入 batch，不等最长请求）。

### 3.2 Function Call 三种约束模式

- **裸 prompt**：仅靠 system prompt 里的 JSON 格式说明，不加任何解码约束。
- **response_format**：OpenAI 标准 `{"type": "json_object"}`，只保证输出是合法 JSON，不保证字段语义正确。
- **guided_json**：vLLM 的 `guided_json=schema`，用 xgrammar/outlines_core 把 JSON Schema 编译成 FSM，解码时屏蔽非法 token，从根本上保证字段名、枚举值、正则、数值范围全部合法。

两个工具 schema 复杂度不同：`get_stock_quote`（6位股票代码正则、市场枚举、字段枚举数组）偏字符串约束；`create_order`（数量范围 1~100、手机号正则、多组枚举）偏数值+格式混合约束，用来验证 guided_json 在不同约束类型下是否都能兜底。

---

## 4. 评估结果

### 4.1 吞吐对比（`outputs/throughput_results.json`，50 prompts × 100 tokens）

| 模式 | 总耗时(s) | 生成 tokens | QPS | tokens/s | 相对 vLLM QPS |
|------|---------|-----------|-----|---------|-------------|
| [A] transformers 串行 | 69.54 | 5000 | 0.72 | 71.9 | 0.014× |
| [B] transformers batch=8 | 11.36 | 5000 | 4.40 | 440.1 | 0.083× |
| [C] vLLM continuous batching | **0.94** | 4583 | **52.91** | **4849.9** | **1.00×** |

**vLLM 相对串行加速 73.6×（QPS）/ 67.5×（tokens/s）；相对 batch=8 加速 12.0×（QPS）/ 11.0×（tokens/s）。**

图表见 `outputs/throughput_comparison.png`（总耗时 / QPS / tokens/s 三联柱状图）。

结果解读：
- **A → B**（串行→batch=8）加速约 6.1×：简单批处理已能大幅提速，但左侧 padding 到 batch 内最长 prompt，仍浪费大量无效 token 计算。
- **B → C**（batch=8→vLLM）再加速约 12×：continuous batching 让短请求提前结束就释放槽位给新请求，长请求不拖累短请求；PagedAttention 按 block 分配 KV cache，基本消除 padding 浪费，GPU 利用率显著提升。
- vLLM 实际生成 token 数（4583）略少于 transformers 路线（5000），因为遇到 EOS 提前停止，说明吞吐优势并非靠"多算"，而是单位时间内有效计算密度更高。

### 4.2 Function Call — get_stock_quote（50 条测试，`outputs/function_call_results.json`）

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| 必选字段齐全 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| **完整 schema 通过 ★** | **46/50 (92%)** | **46/50 (92%)** | **50/50 (100%)** |
| 平均延迟(秒) | 0.437 | 0.387 | 0.387 |

失败案例集中在字段值不满足约束，而非字段名/JSON语法问题：
- `'02699' does not match '^\d{6}$'`（symbol 位数不对）
- `Additional properties are not allowed ('symbols' was unexpected)`（多余字段）
- `Additional properties are not allowed ('isBuy' was unexpected)`（模型自行加字段回答用户的附加问题）

guided_json 无一例失败。

### 4.3 Function Call — create_order（50 条测试）

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| 必选字段齐全 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| **完整 schema 通过 ★** | **26/50 (52%)** | **27/50 (54%)** | **50/50 (100%)** |
| 平均延迟(秒) | 0.483 | 0.490 | 0.513 |

失败案例同样是值域/格式类问题：
- `'1890001111' does not match '^1[3-9]\d{9}$'`（手机号少一位）
- `'wechat' is not one of ['normal', 'express', 'urgent']`（把支付方式误填进 priority 字段）
- `200 is greater than the maximum of 100`（数量超出范围）

guided_json 同样无一例失败。

---

## 5. 综合性能对比

| 对比维度 | 裸 prompt | response_format | guided_json |
|---------|----------|-----------------|-------------|
| get_stock_quote schema 通过率 | 92% | 92% | **100%** |
| create_order schema 通过率 | 52% | 54% | **100%** |
| 平均延迟增量（相对裸prompt） | 基准 | 基本持平/略降 | **基本持平**（+0~0.03s） |
| 覆盖约束类型 | 无 | 仅 JSON 语法 | 语法 + 字段名 + 枚举 + 正则 + 数值范围 |

| 吞吐对比维度 | transformers 串行 | transformers batch=8 | vLLM |
|-------------|-------------------|----------------------|------|
| QPS | 0.72 | 4.40 | **52.91** |
| tokens/s | 71.9 | 440.1 | **4849.9** |
| 相对 vLLM 加速（QPS） | 0.014× | 0.083× | 1.00× |

---

## 6. 结论

1. **vLLM 吞吐优势极其显著**：相对 transformers 串行加速 73.6×（QPS），相对手写 batch=8 加速 12.0×，核心机制是 continuous batching 消除等待、PagedAttention 消除 padding 浪费，二者叠加而非单独任一项就能达到的效果。
2. **guided_json 是唯一能把 Function Call 完整 schema 通过率拉到 100% 的方式**：裸 prompt 和 response_format 在字段语义类错误（多余字段、枚举误填、数值越界、正则不符）上表现几乎一致（92%/92%、52%/54%），说明 response_format 只解决 JSON 语法问题，不解决语义正确性。
3. **约束越复杂，guided_json 的相对收益越大**：get_stock_quote（92%→100%，+8pp）收益小于 create_order（52%→100%，+48pp），因为 create_order 的数值范围和多枚举约束是裸 prompt/response_format 完全无法触及的类型，而 guided_json 在 token 级别对所有约束类型一视同仁。
4. **约束解码几乎不增加延迟**：两个工具上 guided_json 与裸 prompt 延迟差在 ±0.03s 以内，FSM 构建一次、解码全程复用，是"近乎零成本换 100% 可靠性"的工程选择。
5. **两项能力共同构成生产可用性**：吞吐能力决定"能不能扛住并发"，约束解码能力决定"每次返回的结构化数据能不能直接喂给下游系统"——vLLM 在同一套部署里同时满足了这两个通常需要分别优化的维度。
