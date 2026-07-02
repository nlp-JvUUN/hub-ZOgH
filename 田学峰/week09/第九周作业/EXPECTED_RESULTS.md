# RESULTS

## 一、实验环境与前置条件（代码隐含）

| 项目 | 代码要求 |
|------|---------|
| 模型 | `pretrain_models/Qwen2-0.5B-Instruct`，对外名称 `qwen2-0.5b` |
| 推理服务 | vLLM 0.9.2 OpenAI 兼容 server，监听 `0.0.0.0:8000` |
| 客户端 | `openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")` |
| GPU | NVIDIA CUDA 显卡，显存 ≥ 8GB |
| 关键参数 | `max_model_len=2048`, `gpu_memory_utilization=0.6`, `dtype=float16`, `enforce_eager` |

---

## 二、bench_throughput.py — 吞吐对比

### 2.1 测试配置

```python
N_PROMPTS = 50
MAX_NEW_TOKENS = 100
BATCH_SIZE = 8
```

Prompt 组成（代码第 42-65 行）：
- 短问题 × 3 轮 = 30 条（10 个短问题重复 3 次）
- 中等问题 × 1 轮 = 10 条
- 长问题 × 2 轮 = 10 条（5 个长问题重复 2 次）
- 截取前 50 条，最终为：30 短 + 10 中 + 10 长

### 2.2 三路执行逻辑

| 路线 | 实现方式 | 代码位置 |
|------|---------|---------|
| A: transformers serial | `for` 循环逐条 `model.generate()` | 第 97-112 行 |
| B: transformers batch=8 | 每 8 条手动 padding 后 batch generate | 第 114-132 行 |
| C: vLLM | `llm.generate(chat_prompts, SamplingParams(...))` 一次性处理 50 条 | 第 176-180 行 |

### 2.3 预期结果

基于代码逻辑可确定：

1. **vLLM 必然最快**：因为 C 路线一次性接收全部 50 条 prompt，由 vLLM 内部做 continuous batching 和 PagedAttention；而 A/B 路线是手动循环或固定 batch。
2. **serial 必然最慢**：无 batching，每条请求都要等上一条完全结束。
3. **batch=8 居中**：比 serial 快，但存在 padding 浪费和"等最长请求"问题，无法达到理想 8×。

**输出指标预期**：

| 指标 | A serial | B batch=8 | C vLLM |
|------|----------|-----------|--------|
| 总耗时 | 最长 | 中等 | 最短 |
| QPS | 最低 | 中等 | 最高 |
| tokens/s | 最低 | 中等 | 最高 |
| 相对 vLLM 倍率 | < 0.1× | 约 0.05-0.2× | 1.0× |

**代码输出位置**：
- JSON：`outputs/throughput_results.json`
- 图片：`outputs/throughput_comparison.png`

---

## 三、demo_guided_choice.py — 枚举约束

### 3.1 测试配置

- 意图类别：`["查股价", "查财报", "查新闻", "对比分析", "其他"]`
- 测试用例：12 条（代码第 38-51 行）
- 每个用例调用 2 次：裸 prompt + `guided_choice`

### 3.2 评估逻辑

```python
raw_in = raw_out in INTENT_CHOICES
raw_correct = raw_out == expected
guided_correct = guided_out == expected
```

### 3.3 基于代码的预期结果

| 指标 | 裸 prompt | guided_choice |
|------|----------|---------------|
| 输出合法率（在枚举内）| < 100%（模型可能加标点、解释文字） | **100%** |
| 预测正确率 | 取决于模型语义理解 | 与裸 prompt 相同或略高 |
| 平均延迟 | 接近 | 接近（FSM 构建有极小开销） |

**关键结论**：`guided_choice` 通过 FSM 把非法 token 的 logit 置为 `-inf`，因此输出一定落在 5 个枚举值之一；但模型选哪个枚举值仍由语义能力决定，正确率不会自动变成 100%。

---

## 四、demo_guided_regex.py — 正则约束

### 4.1 测试配置

| 任务 | 正则 | 用例数 | 代码位置 |
|------|------|--------|---------|
| 日期标准化 | `r"\d{4}-\d{2}-\d{2}"` | 6 | 第 25-34 行 |
| 股票代码抽取 | `r"\d{6}"` | 5 | 第 37-45 行 |

每个用例调用 2 次：裸 prompt + `guided_regex`。

### 4.2 评估逻辑

```python
def matches(pattern, text):
    return bool(re.fullmatch(pattern, text))
```

### 4.3 基于代码的预期结果

| 任务 | 裸 prompt 合法率 | guided_regex 合法率 |
|------|-----------------|--------------------|
| 日期标准化 | 低-中等（可能带解释、多余文字） | **100%** |
| 股票代码抽取 | 低-中等（可能带"股票代码："前缀） | **100%** |

**关键结论**：`guided_regex` 保证输出严格 `fullmatch` 指定正则，是"下游有严格解析器"场景的标准解法。

---

## 五、demo_guided_json.py — JSON Schema 约束

### 5.1 测试配置

- Schema：财报三元组 `{company: string, year: integer[2015,2025], metric: enum}`
- 测试用例：9 条（代码第 56-66 行）
- 每个用例调用 3 次：裸 prompt / response_format / guided_json
- 评估 5 层指标：合法 JSON / 字段齐全 / year 范围 / metric 枚举 / jsonschema 完全通过

### 5.2 关键测试用例分析

| 用例 | 难点 | 裸 prompt / response_format 预期 | guided_json 预期 |
|------|------|----------------------------------|------------------|
| "茅台 2020 年利润情况" | "利润" 不在枚举 | 可能输出 `metric: "利润"` 导致 schema 失败 | **强制映射到合法枚举** |
| "隆基绿能 22 年 roe" | 年份简写 + 小写指标 | 可能输出 `year: 22` 或 `metric: "roe"` | 强制 `year: 2022`, `metric: "ROE"` |
| "ICBC 2023 营收" | 英文简称 | 字段值可能正确，但 JSON 格式可能出错 | JSON + schema 双保证 |

### 5.3 基于代码的预期结果

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| 合法 JSON | 中等 | **100%** | **100%** |
| 字段齐全 | 中等 | 高 | **100%** |
| year 在 2015~2025 | 中等 | 中等 | **100%** |
| metric 在枚举内 | 中等 | 中等 | **100%** |
| jsonschema 完全通过 | 低-中等 | 中等 | **100%** |

---

## 六、demo_response_format.py — OpenAI 标准 JSON 模式

### 6.1 测试配置

- 任务：新闻情感分析 JSON `{sentiment, confidence, keywords}`
- 测试用例：5 条（代码第 37-43 行）
- 每个用例调用 2 次：裸 prompt / `response_format={"type":"json_object"}`
- 评估 5 层指标：合法 JSON / 有 sentiment 字段 / sentiment 值合法 / 有 confidence / 有 keywords

### 6.2 基于代码的预期结果

| 指标 | 裸 prompt | response_format=json_object |
|------|----------|-----------------------------|
| 合法 JSON | 可能 < 100% | **100%** |
| 有 sentiment 字段 | 可能漏字段 | 接近 100% |
| sentiment 值合法 | 依赖模型 | 依赖模型 |
| 有 confidence 字段 | 依赖模型 | 接近 100% |
| 有 keywords 字段 | 依赖模型 | 接近 100% |

**关键结论**：`response_format` 只保证输出是合法 JSON 对象，不保证字段名、字段类型、枚举值、正则匹配。这是它与 `guided_json` 的本质区别。

---

## 七、demo_function_call.py — 核心综合演示

### 7.1 测试配置

| 工具 | 用例数 | Schema 约束类型 | 代码位置 |
|------|--------|----------------|---------|
| `get_stock_quote` | 50 | string, enum, regex, array, minItems | 第 40-134 行 |
| `create_order` | 50 | integer 范围, 手机号 regex, 多 enum | 第 141-232 行 |

每个工具 × 50 条 × 3 模式 = 300 次请求。

### 7.2 `get_stock_quote` 关键用例与失败预期

#### 7.2.1 枚举/正则类失败（裸 prompt / response_format 易发生）

| 用例 | 可能错误输出 | 失败原因 |
|------|------------|---------|
| "300750 宁德时代最高价" | `{"fields": ["最高价"]}` | `"最高价"` 不在 `enum: ["open","close","high","low","volume"]` |
| "帮我查六零零零零零这只票" | 中文数字无法解析为 6 位代码 | `symbol` 不符合 `^\d{6}$` |
| "查 600.000 今天开盘" | `symbol` 含小数点 | 正则失败 |

#### 7.2.2 缺字段类失败

| 用例 | 可能错误输出 | 失败原因 |
|------|------------|---------|
| "帮我查茅台今天情况" | 漏 `market` 或 `fields` | `required: ["symbol", "market", "fields"]` |
| "随便查一个" | 无有效字段 | 几乎全失败 |

### 7.3 `create_order` 关键用例与失败预期

#### 7.3.1 数量范围失败

| 用例 | 可能错误输出 | 失败原因 |
|------|------------|---------|
| "给我 200 个鼠标..." | `{"quantity": 200}` | 超过 `maximum: 100` |
| "订 0 个苹果..." | `{"quantity": 0}` | 小于 `minimum: 1` |

#### 7.3.2 手机号正则失败

| 用例 | 可能错误输出 | 失败原因 |
|------|------------|---------|
| "订 1 个 iPad，手机 138-1234-5678" | 含连字符 | 不符合 `^1[3-9]\d{9}$` |
| "3 本书，phone +86 13812345678" | 含 "+86" 前缀 | 正则失败 |
| "1 台电脑，电话：12345678" | 位数不足 | 正则失败 |
| "订 2 瓶酒，手机 19912345678" | 199 开头 | 第二位 9 不在 `[3-9]` |

#### 7.3.3 枚举诱导失败

| 用例 | 可能错误输出 | 失败原因 |
|------|------------|---------|
| "订 1 个保温杯...用银联卡" | `payment_method: "unionpay"` | 不在 `enum: ["alipay","wechat","card"]` |
| "3 本书...cash 付款" | `payment_method: "cash"` | 不在枚举 |
| "10 瓶酒...standard" | `priority: "standard"` | 不在 `enum: ["normal","express","urgent"]` |

### 7.4 基于代码的量化预期

| 工具 | 指标 | 裸 prompt | response_format | guided_json |
|------|------|----------|-----------------|-------------|
| get_stock_quote | JSON 语法合法 | 中等（部分用例会输出解释文字） | **100%** | **100%** |
| get_stock_quote | 必选字段齐全 | 中等 | 高 | **100%** |
| get_stock_quote | **完整 schema 通过** | 低-中等 | 中等 | **100%** |
| create_order | JSON 语法合法 | 中等 | **100%** | **100%** |
| create_order | 必选字段齐全 | 中等 | 高 | **100%** |
| create_order | **完整 schema 通过** | 低 | 低-中等 | **100%** |

**延迟预期**：三种模式平均延迟接近，因为约束解码的 FSM 构建开销会被 vLLM 缓存，且每步 logit mask 计算量很小。

---

## 八、跨脚本核心结论

| 能力 | 裸 prompt | response_format | guided_json / guided_regex / guided_choice |
|------|----------|-----------------|-------------------------------------------|
| JSON 语法合法 | 不稳定 | **100%** | **100%** |
| 字段/值语义正确 | 不稳定 | 不稳定 | **100%**（在 schema/regex/choice 范围内） |
| 跨厂商可移植性 | 高 | **高** | 低（vLLM 扩展） |
| 延迟代价 | 基准 | 接近基准 | 接近基准（FSM 缓存后） |

---

## 九、文件输出预期

执行全部脚本后，代码会在 `outputs/` 目录生成以下文件：

| 文件 | 生成脚本 | 内容 |
|------|---------|------|
| `outputs/throughput_results.json` | bench_throughput.py | 三路 benchmark 的耗时、QPS、tokens/s |
| `outputs/throughput_comparison.png` | bench_throughput.py | 三路对比柱状图 |
| `outputs/function_call_results.json` | demo_function_call.py | 两个工具 × 50 条 × 3 模式的详细结果 + 失败案例 |

