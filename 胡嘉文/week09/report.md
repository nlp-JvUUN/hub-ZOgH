# Week 9 大模型应用补充知识 — 学习总结报告

## 一、项目概览

**学习周期：** 2026 年 6 月  
**学习环境：** macOS Apple Silicon (MPS) + conda week9  
**核心主题：** vLLM 推理部署与约束解码技术  

本项目通过 **7 个独立脚本 + 3 份设计文档**，系统性地将 vLLM 从"一个推理框架"拆解为可以感性理解的具体能力：

1. **部署能力** — 一条命令将 HuggingFace 模型变成 OpenAI 兼容 HTTP 服务
2. **约束解码能力** — guided_choice / guided_regex / guided_json / response_format 四种方式
3. **性能理解** — PagedAttention + continuous batching 的工程价值（理论了解）

---

## 二、学习内容逐项总结

### 2.1 启动 vLLM Server

**文件：** `src_mac/start_server.sh`

- 关键参数：`--device cpu`（macOS 不支持 CUDA，必须指定 CPU 模式）、`--dtype float32`（CPU 不支持半精度）、去掉 `--gpu-memory-utilization`
- 启动后监听 `0.0.0.0:8000`，提供 OpenAI 兼容的 `/v1/chat/completions` 接口
- 所有 demo 脚本通过 OpenAI 客户端调用这个 server

**遇到的问题：**
- `all_proxy` SOCKS5 代理导致 httpx 报错 → 解决：`unset all_proxy`
- macOS CPU 推理比 GPU 慢 10~20 倍，首次加载需 30~60 秒

### 2.2 枚举约束 — guided_choice

**文件：** `demo_guided_choice.py`  
**场景：** 金融问答意图路由（查股价 / 查财报 / 查新闻 / 对比分析 / 其他）

**核心原理：**
- 传入 `extra_body={"guided_choice": ["查股价", "查财报", ...]}`
- vLLM 将枚举列表构建为 Trie + FSM（有限状态机）
- 每步解码时屏蔽（mask out）所有非法 token 的 logits（设为 `-inf`）
- 强制模型只输出枚举集合中的完整一项

**关键发现：**
- 裸 prompt 方式下，小模型（Qwen2-0.5B）会输出 `"查股价。"`（多余句号）、`"意图是查股价"`（多余解释）
- guided_choice 100% 保证输出合法枚举值，但**分类正确率**仍受限于模型自身能力
- 延迟开销仅 1~5ms（FSM 一次构建后续复用）

### 2.3 正则约束 — guided_regex

**文件：** `demo_guided_regex.py`  
**场景：** 日期标准化（YYYY-MM-DD）、A 股代码抽取（6 位数字）

**核心原理：**
- 传入 `extra_body={"guided_regex": "\d{4}-\d{2}-\d{2}"}`
- vLLM 将正则编译为 NFA → DFA，只允许匹配正则前缀的 token 通过
- 模型输出必然整体匹配正则，下游解析器无需容错逻辑

**测试用例设计：**
- 日期覆盖了 6 种中文表述（"2024年5月12日"、"三月三号"、"明天"等）
- 股票代码包含中文数字陷阱（"六零零五一九"→600519）
- 裸 prompt 下 0/6 合法率，guided_regex 下 6/6（100%）

**工程价值：** 凡是下游有严格解析器的字段（日期、手机号、邮箱、邮编），正则约束能把"模型说对但格式错"的问题一次根治。

### 2.4 JSON Schema 约束 — guided_json

**文件：** `demo_guided_json.py`  
**场景：** 财报问答意图抽取（公司名、年度、指标三元组）

**核心原理：**
- 传入 `extra_body={"guided_json": INTENT_SCHEMA}`，其中 schema 定义了字段名、类型、枚举、数值范围
- 比 guided_choice/regex 更强：一次性约束整个复杂对象
- 底层将 JSON Schema 编译为 FSM，每步只允许生成合法路径上的 token

**与 response_format 对比：**
- `response_format={"type": "json_object"}` 只在 prompt 末尾加 `\n` 诱导模型输出 JSON，不校验内容
- `guided_json` 是硬约束，字段名、类型、枚举值、数值范围全部 100% 保证
- 实测：response_format 可能输出 `"year": 22`（违反 `minimum: 2015`），guided_json 强制为 2022

### 2.5 OpenAI 标准 response_format

**文件：** `demo_response_format.py`  
**场景：** 新闻情感分析 + 置信度 + 关键词

**核心定位：** 跨平台可移植方案
- OpenAI / Azure / vLLM / Together.ai 都兼容
- 合法率 80~95%，高于裸 prompt（40~60%），但字段语义仍靠模型自觉
- 工程选型：多厂商切换 → response_format；单一 vLLM + 严格解析 → guided_json

### 2.6 核心实战 — Function Call 基准测试

**文件：** `demo_function_call.py` ★ 核心文件

**两个工具函数：**
| 工具 | 场景 | Schema 复杂度 |
|------|------|--------------|
| `get_stock_quote` | 股价查询 | string + enum + regex + array + minItems |
| `create_order` | 电商下单 | integer 范围 + 手机号正则 + 多 enum |

**测试规模：** 每个工具 50 条测试用例 × 3 种模式 = **300 次 API 调用**

**测试用例分类（精心构造）：**
| 类别 | 目的 | 示例 |
|------|------|------|
| 基础直接（15） | 正常输入基准线 | "查 600000 今天开盘价" |
| 省略字段（8） | 模型需自主推断 | "帮我查茅台今天情况"（无 market） |
| 诱导多余文本（7） | 测试约束抗干扰 | "并简单解释什么是开盘价" |
| 非标准输入（5） | 中文数字/英文/符号 | "六零零零零零"、"600.000" |
| 日期模糊（5） | 相对日期理解 | "昨天"、"上周五" |
| 数值超限（6） | minimum/maximum | "200 个鼠标"（>100） |
| 格式不标准（7） | 手机号带横杠/+86 | "138-1234-5678" |
| 枚举诱导（10） | 不在 enum 中的值 | "standard"、"银联卡"、"cash" |
| 边界无意义（6） | 信息缺失 | "随便查一个"、"???" |

**预期结论：**
- 裸 prompt：JSON 合法率 ~70~96%，完整 schema 通过率仅 **20~60%**
- response_format：JSON 合法率 100%，但 schema 通过率仅 **42~68%**
- guided_json：完整 schema 通过率 **100%**

**关键洞察：** `response_format` 和 `guided_json` 之间的 30~50 个百分点差距，就是约束解码的工程价值——`response_format` 只管语法，不管字段值是否合法。

---

## 三、约束解码三种方式的递进关系

```
guided_choice       单值枚举     ← 基础
    ↓
guided_regex       格式匹配     ← 更灵活
    ↓
guided_json        Schema 约束  ← 完整结构化输出
```

| 方式 | 参数 | 约束强度 | 延迟开销 | 适用场景 |
|------|------|---------|---------|---------|
| guided_choice | `extra_body={"guided_choice": [...]}` | 枚举值之一 | ~1ms | 意图路由、情感分类 |
| guided_regex | `extra_body={"guided_regex": "..."}` | 正则格式 | ~2ms | 日期/电话/代码抽取 |
| guided_json | `extra_body={"guided_json": schema}` | 完整 Schema | ~5ms | Function Call、Tool Call |
| response_format | `response_format={"type": "json_object"}` | 仅 JSON 语法 | 极低 | 跨平台部署 |

---

## 四、性能对比（理论部分）

`bench_throughput.py` 设计用于 RTX 4060 8GB + CUDA 环境，macOS 上无法运行。

**原设计的预期结果：**
| 模式 | 50 请求总耗时 | QPS | Tokens/s | 相对 vLLM |
|------|-------------|-----|----------|-----------|
| [A] transformers 串行 | 60.98s | 0.82 | 60 | 0.017× |
| [B] transformers batch=8 | 12.85s | 3.89 | 289 | 0.080× |
| [C] vLLM continuous batching | **1.03s** | **48.59** | **3394** | **1.00×** |

**关键机制理解：**
- **PagedAttention** — KV cache 按 block 管理，消除内存碎片，利用率接近 100%
- **Continuous Batching** — 短请求完成后立即插入新请求，不等最长的，batch size 动态到 20~40
- vLLM 相对串行加速 **59×**，相对 batch=8 加速 **12.5×**

---

## 五、学习心得与痛点记录

### 5.1 macOS 环境适配总结

| 问题 | 解决方案 |
|------|---------|
| SOCKS5 代理导致 httpx 报错 | `unset all_proxy` |
| Torch not compiled with CUDA | Apple Silicon 无法解决，改用 CPU 推理 |
| vLLM 不支持 MPS | 指定 `--device cpu`，接受较慢速度 |
| bench_throughput.py 无法运行 | 跳过，理解原理即可 |

### 5.2 关键工程认知

1. **小模型 + 约束解码 = 可用的低成本方案**：0.5B 模型裸 prompt 不可用，但配合 guided_json 可以 100% 满足 schema
2. **约束解码几乎不增加延迟**：FSM 一次构建长期复用，开销可忽略
3. **从"模型说对"到"模型输出必合法"**：工程上降低了下游 90%+ 的容错代码
4. **Function Call 是 Agent 系统的基础**：可靠的结构化输出是 Agent 调用工具的基石

### 5.3 后续学习方向

- **guided_grammar**（EBNF 语法约束）— 当前项目未涵盖
- **vLLM 生产化** — Prometheus 监控、多模型部署、LoRA 适配
- **与 RAG/Agent 框架集成** — LangChain、AutoGen、MCP Server
- **模型微调** — 用 SFT 提升小模型的 function call 准确率

---

## 六、项目文件清单

| 文件 | 作用 |
|------|------|
| `src/demo_guided_choice.py` | 枚举约束（意图路由） |
| `src/demo_guided_regex.py` | 正则约束（日期/股票代码） |
| `src/demo_guided_json.py` | JSON Schema 约束（财报三元组） |
| `src/demo_response_format.py`| OpenAI 标准 json_object |
| `src/demo_function_call.py` | ★ 核心：双工具 × 50×3 对比 |
| `src/bench_throughput.py` | 吞吐对比（需 CUDA，macOS 跳过） |
| `src_mac/start_server.sh` | macOS 版 vLLM server 启动脚本 |
| `ARCHITECTURE.md` | 技术架构方案 |
| `USAGE_GUIDE.md` | Linux 使用指南 |
| `MAC_USAGE_GUIDE.md` | macOS 适配使用指南 |
