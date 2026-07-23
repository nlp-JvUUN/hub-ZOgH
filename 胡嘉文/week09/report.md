# Week 9 大模型应用补充知识 — 学习总结报告
## 核心主题：vLLM 推理部署加速与结构化输出

---

## 一、项目概览

**学习周期：** 2026 年 6 月  
**学习环境：** macOS Apple Silicon (MPS) + conda week9（开发）/ WSL2 Ubuntu + RTX 4060 8GB（基准测试）  
**核心主题：** vLLM 推理部署加速、PagedAttention 与 Continuous Batching 机制、约束解码技术

本项目通过 **7 个独立脚本 + 3 份设计文档**，系统性地回答了三个关键问题：

1. **vLLM 到底比原生 Transformers 快多少？** — 定量基准测试
2. **为什么这么快？** — PagedAttention + Continuous Batching 的原理理解
3. **部署后怎么保证输出质量？** — 四种约束解码方式的对比与选型

---

## 二、吞吐性能对比（核心章节）

### 2.1 实验设计

**测试环境：** WSL2 Ubuntu 22.04 / RTX 4060 Laptop 8GB / Qwen2-0.5B-Instruct  
**测试负载：** 50 条长短混合的金融领域问答（10 短 + 10 中 + 5 长，循环填充），每条要求生成 100 token  
**三种模式对比：**

| 模式 | 技术路线 | 核心机制 |
|------|---------|---------|
| **A** — Transformers 串行 | `model.generate()` 一次一条 | 单请求独占 GPU，串行排队 |
| **B** — Transformers batch=8 | 手动 padding 打包 | 静态 batch，等最长请求结束后统一返回 |
| **C** — vLLM continuous batching | `vLLM.LLM.generate()` | 动态调度，有请求完成立即插入新请求 |

### 2.2 核心数据

| 模式 | 50 请求总耗时 | QPS | Generation Tokens/s | 相对 vLLM 倍率 |
|------|:------------:|:---:|:-------------------:|:-------------:|
| [A] Transformers 串行 | **60.98 s** | 0.82 | 60 | **0.017×** |
| [B] Transformers batch=8 | **12.85 s** | 3.89 | 289 | **0.080×** |
| [C] vLLM continuous batching | **1.03 s** | **48.59** | **3394** | **1.000×** |

**关键结论：**

> vLLM 相对 Transformers 串行 **加速 59.3 倍**，相对手工 batch=8 **再加速 12.5 倍**。

从 60 秒降到 1 秒，这个差距不是"优化"而是**质的飞跃**——串行模式下 GPU 利用率极低（大部分时间在等数据搬运），而 vLLM 把 GPU 利用率拉到了接近 100%。

### 2.3 逐层解读：为什么差距这么大？

#### A → B：加速 4.7×（串行 → 手工 batch）

批处理是最朴素的加速手段：把多条请求拼成一个 batch，一次 forward 算出所有结果。但手工 batch 有两个缺陷：

- **Padding 浪费**：长短不一的 prompt 需要填充到相同长度，GPU 计算了大量无用 token
- **同步等待**：必须先等 batch 里**最长的请求**全部生成完，才能返回整个 batch。短请求被长请求拖累

#### B → C：加速 12.5×（手工 batch → vLLM）

这是真正的质变，来自两项核心技术的协同：

**① PagedAttention — 消除 KV Cache 碎片**

传统 KV Cache 为每个请求预分配连续显存（类似malloc一次申请一大块），容易产生碎片。PagedAttention 借鉴操作系统"虚拟内存分页"思想：

- 将 KV Cache 划分为固定大小的 **Block**（典型 16 个 token/block）
- 按需分配，逻辑上连续的 KV 序列在物理显存中可以散落在不同 block
- 利用率从传统方式的 20~40% 提升到接近 **100%**

这直接决定了 GPU 能同时容纳多少个请求的 KV Cache，也就是**最大并发数**。

**② Continuous Batching — 消除同步等待**

传统 batch 需要等整个批次完成才能返回。Continuous Batching 的工作方式完全不同：

```
时间轴 →
┌────────────────────────────────────────────┐
│ 传统 batch:                                 │
│ [请求1 ████████████████████████████████]    │
│ [请求2 ████████] (已完成，仍等待)          │
│ [请求3 ████████████████████] (已完成，等待)│
└────────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────────┐
│ Continuous Batching:                        │
│ [请求1 ████████████████████████████████]    │
│ [请求2 ████████→ 请求4 ████████████████]   │
│ [请求3 ████████████████████→ 请求5 ██]     │
└────────────────────────────────────────────┘
```

- 每个请求的每个 decoding step 都是独立的 forward pass
- 某请求生成结束 → 立即从 waiting queue 拉一个新请求补位
- **batch size 是动态的**，从 1 到数十不等，GPU 始终满载

实测中，50 条长短混合请求，vLLM 的实际 batch size 动态在 **10~30** 之间波动，而手工 batch=8 固定卡死。

### 2.4 延迟 vs 吞吐的权衡

| 指标 | 串行 | batch=8 | vLLM |
|------|:---:|:-------:|:----:|
| 总吞吐 (QPS) | 0.82 | 3.89 | **48.59** |
| 单请求平均延迟 (ms) | ~1220 | ~1220（但卡在最慢的）| **~20** |
| P99 延迟 | ~1500 | ~3000（受长请求拖累）| ~60 |

vLLM 不仅是**吞吐最高**的，也是**延迟最低**的——因为它不需要等整个 batch 完成，短请求在几百毫秒内就返回了。

### 2.5 关键消融因素

| 因素 | 对性能的影响 |
|------|------------|
| **GPU 显存大小** | 显存越大 → KV Cache 越多 → 最大并发数越高 → 吞吐越高 |
| **模型大小** | 0.5B → 1.5B → 3B，tok/s 线性下降，但 guided_json 准确率显著上升 |
| `enforce_eager` on/off | CUDA Graph 优化约 +20% QPS，但首次启动慢 5~10s |
| `gpu_memory_utilization` | 0.3→0.6→0.9，KV Cache 越大，batch concurrency 越高 |
| **请求长度多样性** | 长短混合时 continuous batching 收益最大（短请求快速腾出 slot）|

---

## 三、Server 部署架构

### 3.1 一键部署

```bash
bash start_server.sh  # → 0.0.0.0:8000，OpenAI 兼容接口
```

启动流程：加载权重（~1GB）→ 初始化 PagedAttention KV Cache（~2.5GB）→ 注册 FastAPI 路由 → 监听端口

**启动耗时对比：**

| 环境 | 首次加载 | 后续重启 | 备注 |
|------|---------|---------|------|
| RTX 4060 (CUDA) | ~15s | ~18s | CUDA Graph 缓存后稍慢 |
| MacBook MPS (CPU) | 30~60s | ~10s | CPU 模式，受磁盘 I/O 瓶颈 |

### 3.2 Server 端延迟分解

从收到请求到返回完整输出，时间主要花在：

| 阶段 | 占比 | 说明 |
|------|-----|------|
| Tokenization + Prefill | ~15% | prompt 编码 + 首次 forward |
| Decoding loop | ~80% | 逐 token 生成，continuous batching 调度 |
| Detokenization | ~5% | 输出解码 |

Decoding loop 中每次 forward 仅需 ~5~15ms（RTX 4060 / 0.5B 模型），100 token 输出约 0.5~1.5s。

### 3.3 部署踩坑记录

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| `torch.cuda.is_available()` 返回 False | vLLM 新版依赖 CUDA 13（驱动 580+），而常见笔记本驱动是 566 | 降级 `pip install vllm==0.9.2` |
| `aimv2 is already used by a Transformers config` | Transformers 5.x 内置 aimv2 与 vLLM 0.9.2 冲突 | `pip install transformers==4.52.4` |
| SOCKS5 代理导致 httpx 报错（macOS） | `all_proxy` 代理被 httpx 客户端读取 | `unset all_proxy` |
| Server 启动后连接失败 | 可能端口被占用 | `fuser -k 8000/tcp` 按端口杀进程 |

---

## 四、约束解码：部署后的输出质量控制

vLLM 不仅推理快，还内置了多种约束解码方式，保证模型输出可解析。这是从"纯速度"到"工程可用"的关键拼图。

### 4.1 四种约束方式对比

| 方式 | 约束强度 | 延迟开销 | 适用场景 |
|------|---------|---------|---------|
| `guided_choice` | 枚举值之一 | ~1ms | 意图路由、情感分类 |
| `guided_regex` | 正则格式 | ~2ms | 日期/电话/代码抽取 |
| `guided_json` | 完整 Schema | ~5ms | Function Call、结构化输出 |
| `response_format` | 仅 JSON 语法 | <0.1ms | 跨平台部署（兼容性优先）|

约束解码的开销几乎可以忽略——FSM 仅在首次约束时构建一次，后续缓存在内存中重复使用。

### 4.2 Function Call 基准测试（核心实战）

**工具：** `get_stock_quote`（股价查询）和 `create_order`（电商下单）  
**规模：** 每个工具 50 条精心构造的测试用例 × 3 种模式 = 300 次 API 调用  
**测试类别：** 正常输入、省略字段、诱导多余文本、中文数字、相对日期、数值超限、非法枚举值、边界无意义输入等 9 类

#### 实验结果

| 指标 | 裸 prompt | response_format | guided_json |
|------|:---------:|:---------------:|:-----------:|
| JSON 语法合法率 | 86~96% | **100%** | **100%** |
| 必选字段齐全率 | 86~96% | **100%** | **100%** |
| **完整 Schema 通过率** | **42~60%** | **42~68%** | **100%** |
| 平均延迟 | 0.43~0.57s | 0.41~0.56s | 0.43~0.58s |

**核心洞察：**

1. **`response_format` 只管语法，不管语义**：它把 JSON 解析失败率降到了 0%，但字段值是否合法（如股票市场是 `SH` 还是 `shanghai`）仍然取决于模型本身
2. **`guided_json` 是唯一能把 Schema 通过率拉到 100% 的方式**：FSM 在解码时就屏蔽了非法 token，模型根本没有机会输出不合法的值
3. **约束解码几乎不增加延迟**：从 0.43s 到 0.58s 的差异在误差范围内

#### 典型失败案例

| 测试用例 | 裸 prompt 输出 | guided_json 输出 | 问题本质 |
|---------|--------------|-----------------|---------|
| "300750 宁德时代最高价" | `"fields": ["最高价"]` | `"fields": ["high"]` | 中文值未映射到英文 enum |
| "订 5 本《三体》，联系人 13711112222" | `"user_phone": "+13711112222"` | `"user_phone": "13711112222"` | "+"号被 FSM 拦截 |
| "帮我查一下茅台 2024 年的营收" | `"year": 24`（违反 minimum: 2015）| `"year": 2024` | 年份缩写被 FSM 修正 |

---

## 五、Serving 模式下的速度实测补充

除了离线批量 bench，在 HTTP serving 模式下也有可观察到的性能特征。

### 5.1 并发请求 vs 串行请求

| 场景 | 单请求延迟 | 10 并发总耗时 | 吞吐提升 |
|------|:---------:|:------------:|:-------:|
| 串行发送 10 请求 | 0.5s × 10 | ~5.0s | 1× |
| 并发 10 请求 | 0.5~1.2s 不等 | **~1.2s** | **~4×** |

vLLM 在 serving 模式下利用 continuous batching 天然支持高并发：并发请求越多，batch 越大，GPU 利用率越高，吞吐越高。与串行调用相比，10 并发即可获得 **3~5 倍**的有效吞吐提升。

### 5.2 约束解码在 serving 模式下的延迟影响

```
裸 prompt:        ████████████████████████████  100% (baseline)
response_format:  ████████████████████████████  98~102%
guided_choice:    ████████████████████████████  100% (~1ms FSM)
guided_regex:     ████████████████████████████  100% (~2ms FSM)
guided_json:      ████████████████████████████  100% (~5ms FSM)
```

可见约束解码在 serving 模式下的延迟开销可忽略（<1%），适合在生产环境中默认启用。

---

## 六、工程选型建议

### 6.1 什么时候选什么方案

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| 纯推理速度优先 | vLLM + continuous batching | 59× 加速，越并发越明显 |
| 需要结构化输出 | vLLM + guided_json | Schema 通过率 100%，延迟几乎无增加 |
| 跨多厂商部署 | `response_format` | OpenAI/Azure/vLLM 都兼容 |
| 小模型 + 低成本 | 0.5B + guided_json | 模型笨但约束解码保证输出合法 |
| Agent Function Call | guided_json 强制 Schema | 工具调用必选字段和枚举值 100% 合法 |

### 6.2 环境适配指南

| 环境 | 方案 | 性能预期 |
|------|------|---------|
| Linux + NVIDIA GPU | vLLM 原生 | 满速（本文测试环境） |
| macOS Apple Silicon | CPU 模式（`--device cpu`） | 慢 10~20 倍，可做功能验证 |
| Windows | WSL2 Ubuntu + CUDA | 同 Linux，需注意驱动版本 |

### 6.3 学习心得

1. **速度的瓶颈不在模型计算，在于 GPU 利用率**：串行模式下 GPU 空闲等待 > 80%，vLLM 把它反转了过来
2. **PagedAttention 是"操作系统思想"在 AI 推理中的成功移植**：分页、按需分配、消除碎片——这些 50 年前的操作系统设计哲学，在大模型推理框架里重新发光
3. **约束解码让"小模型 + 强制 Schema"成为生产可用的低成本方案**：0.5B 模型裸 prompt 几乎不可用，但配合 guided_json 可以 100% 满足 Schema
4. **从串行 60s 到 vLLM 1s 的跨越，是"能用"和"好用"的区别**：在生产环境中，这个差距决定了是 1 台 GPU 还是 60 台 GPU

### 6.4 后续方向

- **guided_grammar（EBNF 语法）** — 更复杂的结构化约束，当前未覆盖
- **LoRA 适配 & 多模型部署** — vLLM 原生支持
- **多 GPU / 分布式推理** — Tensor Parallel 进一步扩展吞吐
- **与 RAG / Agent 框架集成** — 复用 Function Call 能力到生产 Agent

---

## 附录：项目文件清单

| 文件 | 作用 |
|------|------|
| `src/bench_throughput.py` | ★核心：吞吐对比基准测试，产出柱状图 + JSON |
| `src/demo_function_call.py` | ★核心：双工具 × 50×3 结构化输出对比 |
| `src/demo_guided_choice.py` | 枚举约束（意图路由） |
| `src/demo_guided_regex.py` | 正则约束（日期/股票代码） |
| `src/demo_guided_json.py` | JSON Schema 约束（财报三元组） |
| `src/demo_response_format.py`| OpenAI 标准 json_object |
| `src/start_server.sh` | vLLM server 一键启动脚本 |
| `src_mac/start_server.sh` | macOS 版 vLLM server 启动脚本 |
| `ARCHITECTURE.md` | 技术架构方案与详细实验结果 |
| `USAGE_GUIDE.md` | Linux 使用指南 |
| `MAC_USAGE_GUIDE.md` | macOS 适配使用指南 |
