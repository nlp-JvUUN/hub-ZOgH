# 水下机器人专利 RAG 问答系统

> 基于检索增强生成（RAG）的水下机器人专利技术智能问答系统。数据覆盖 AUV/ROV/水下无人机/声学通信/海底测绘/推进系统/仿生机器人等方向，中英文混合（US + CN + PCT），共 21 份专利。

---

## 目录

- [项目定位](#项目定位)
- [系统架构与数据流](#系统架构与数据流)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [与 week10_rag（上市公司年报）对比](#与-week10_rag上市公司年报-对比)
- [评测题集](#评测题集)
- [踩坑与注意事项](#踩坑与注意事项)
- [目录结构](#目录结构)
- [扩展方向](#扩展方向)

---

## 项目定位

本项目是 `week10_rag`（上市公司年报 RAG）的姊妹项目，**共用同一套代码架构**，但数据源从"巨潮资讯网 PDF 年报"切换为"水下机器人专利数据"。

| 维度 | `week10_rag` 年报版 | `week10_patent_rag` 专利版 |
|------|-------------------|--------------------------|
| 数据源 | 巨潮资讯网 PDF（~85MB） | 本地生成专利数据集 |
| 解析复杂度 | 高（pdfplumber + PyMuPDF + OCR，~300行） | 低（JSON 字段映射，~100行） |
| 分块核心差异 | 财务表格单独成块 | 权利要求（claims）逐条独立成块 |
| 检索特点 | 财务数字精确匹配 | 技术术语精确匹配（BM25 优势更显著） |
| 跨文档挑战 | 同公司多年数据对比 | 不同专利权人技术路线对比 |
| 代码行数 | ~700 行 | ~600 行（无需 LangChain 版） |

**设计理念**：通过两个项目的代码对比，直观展示"数据源质量决定工程复杂度"这一核心工程认知。

---

## 系统架构与数据流

### 流水线总览

```
专利数据集（内置 21 份）
      │
      ▼  src/fetch_patents.py
  data/raw/  (21 个 JSON，含完整专利文本)
      │
      ▼  src/parse_patents.py
  data/parsed/  (21 个 JSON，含结构化 blocks)
      │
      ▼  src/chunk_documents.py
  data/chunks/  (all_semantic.json，233 个 chunks)
      │
      ▼  src/build_index.py
  vectorstore/faiss_index.bin + faiss_meta.json  (932 KB)
      │
      ▼  src/rag_pipeline.py  (命令行) / src/serve.py  (HTTP 服务)
  用户查询 → 向量检索 + BM25 → RRF 融合 → (Rerank) → LLM 生成 → 答案 + 引用
      │
      ▼  evaluation/
  20 道评测题
```

### 各阶段数据形态变化

```
原始专利 JSON
  → ParsedBlock (title/abstract/claim/text + 章节路径)
    → Chunk (语义段落，含专利ID/权利人/权利要求编号)
      → Vector (1024维 float32，L2归一化)
        → Top-10 (余弦相似度 + BM25 → RRF 融合)
          → Top-4 → LLM → 答案
```

### 检索架构

```
用户查询
  ├─ (可选) 查询改写: qwen-turbo
  ├─ 向量检索: DashScope text-embedding-v3 + FAISS IndexFlatIP
  ├─ BM25 检索: jieba 分词 + rank_bm25
  ├─ RRF 融合: score(d) = Σ 1/(60 + rank_i(d))
  ├─ (可选) CrossEncoder Rerank: bge-reranker-base
  ├─ 相关性阈值: vec_score < 0.25 → 拒绝回答
  └─ LLM 生成: DashScope qwen-plus, temperature=0.1
```

### 专利特有的分块策略

| block_type | 处理方式 | 原因 |
|-----------|---------|------|
| `title` | 独立成块 | 标题是检索的高质量入口 |
| `abstract` | 独立成块 | 摘要浓缩了技术方案的核心 |
| `claim` | **每条单独成块** | 每条权利要求是独立的法律主张，不能切断 |
| `text` | 段落累积到 800 字符 | 说明书内容，保持语义完整 |

---

## 快速开始

### 环境要求

- Python 3.11+（推荐 conda 环境 `llm311`）
- DashScope API Key（阿里云，用于 embedding 和 LLM 生成）

### 安装依赖

```powershell
pip install -r requirements.txt
```

核心依赖：`openai` `faiss-cpu` `rank_bm25` `jieba` `numpy` `fastapi` `uvicorn`

### 配置 API Key

```powershell
# 临时设置（当前终端有效）
$env:DASHSCOPE_API_KEY = "sk-你的key"

# 永久设置（Windows，推荐）
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-你的key", "User")
```

> **注意**：通过 `[Environment]::SetEnvironmentVariable` 设置后，需要**重新打开终端**才会生效。如果旧终端读不到，可临时用 `$env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")` 加载。

### 一键执行（5 步流水线）

```powershell
cd D:\mydocs\workspace\llm_demo1\scratch-2026\week10_patent_rag

# Step 1: 生成专利数据集（无需网络，内置数据）
python src/fetch_patents.py
# 输出: data/raw/ (21 个 JSON) + data/manifest.json

# Step 2: 解析专利文本 → 结构化 blocks
python src/parse_patents.py
# 输出: data/parsed/ (21 个 JSON, 共 266 blocks)

# Step 3: 文档分块
python src/chunk_documents.py
# 输出: data/chunks/all_semantic.json (233 个 chunks)

# Step 4: 构建向量索引（调用 DashScope API，约 10 秒）
python src/build_index.py
# 输出: vectorstore/faiss_index.bin (932 KB) + faiss_meta.json

# Step 5: 启动问答
python src/rag_pipeline.py --query "AUV水下避障技术有哪些主流方案？"
```

### 实测数据

| 步骤 | 产物 | 数量 | 耗时 |
|------|------|------|------|
| fetch | 原始 JSON | 21 份专利 | < 1 秒 |
| parse | 结构化 blocks | 266 个 (title=21, abstract=21, claim=117, text=107) | < 1 秒 |
| chunk | 语义分块 | 233 个 (平均 351 字符) | < 1 秒 |
| build | FAISS 索引 | 233 × 1024 维 (932 KB) | ~10 秒 (24 批 API 调用) |
| query | 单次问答 | 1 个答案 + 4 个引用 | 3~8 秒 (2 次 API 调用) |

---

## 使用方式

### 命令行问答

```powershell
# 交互式（连续多轮问答）
python src/rag_pipeline.py

# 单次查询
python src/rag_pipeline.py --query "水下声学通信的主要技术挑战是什么？"

# 按专利权人过滤
python src/rag_pipeline.py --query "水下推进系统" --assignee "Blue Robotics"

# 按专利局过滤
python src/rag_pipeline.py --query "SLAM" --patent-office "CNIPA (中国)"

# 开启查询改写（将模糊问题优化为检索关键词）
python src/rag_pipeline.py --query "水下机器人怎么导航的" --query-rewrite

# 消融测试：关闭 BM25 / Rerank
python src/rag_pipeline.py --query "..." --no-bm25
python src/rag_pipeline.py --query "..." --no-rerank
```

### 交互模式特殊命令

- `exit` — 退出
- `mode` — 查看当前 BM25 / Rerank / QueryRewrite 开关状态

### HTTP 服务

```powershell
cd src
uvicorn serve:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000` 进入**教学可视化页面**（深海蓝暗色主题），可逐步查看检索流水线的每一步中间结果：

```
① 向量检索 Top-5 → ② BM25 检索 Top-5 → ③ RRF 融合 Top-5 → ④ LLM 上下文 Top-4 → ⑤ 生成答案 + 引用
```

API 接口（Swagger 文档: `http://127.0.0.1:8000/docs`）：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 标准问答，返回答案 + 引用 |
| `/query/debug` | POST | 教学调试，返回每步中间结果 |
| `/health` | GET | 健康检查 |

---

## 与 week10_rag（上市公司年报）对比

### 代码层面

| 文件 | 年报版 | 专利版 | 改动程度 |
|------|-------|--------|---------|
| 数据获取 | `download_reports.py` (巨潮 API, ~190行) | `fetch_patents.py` (内置数据集, ~120行) | **全新** |
| 数据解析 | `parse_pdf.py` (pdfplumber+PyMuPDF+OCR, ~330行) | `parse_patents.py` (JSON映射, ~190行) | **全新**（代码量仅 58%） |
| 文档分块 | `chunk_documents.py`（~290行） | `chunk_documents.py`（~210行） | **5%**（新增 claim/abstract 类型） |
| 建索引 | `build_index.py`（~222行） | `build_index.py`（~160行） | **5%**（元数据字段适配） |
| 问答流水线 | `rag_pipeline.py`（~430行） | `rag_pipeline.py`（~330行） | **5%**（System prompt 适配） |
| Web 服务 | `serve.py`（~272行） | `serve.py`（~220行） | **5%**（路径、文案） |
| 前端页面 | `index.html`（浅色主题） | `index.html`（深海蓝暗色主题） | **改皮** |

> 核心发现：**从分块到生成的核心管道几乎原封不动**，验证了这套 RAG 架构的领域通用性。

### 数据源质量对比

| 维度 | 年报 PDF | 专利 JSON |
|------|---------|----------|
| 解析所需库 | 3 个 (pdfplumber + PyMuPDF + Tesseract) | 0 个 (纯 Python) |
| 结构噪声 | 页眉页脚、扫描件、嵌套表格 | 无 |
| 章节识别 | 靠字体大小/加粗猜测 | 天然字段分隔 |
| 表格处理 | pdfplumber 规则引擎 + Markdown 转换 | 无表格（专利以文字描述为主） |
| OCR 降级 | 必需（审计报告原件为扫描件） | 不需要 |

**结论**：年报版的 1/3 工程复杂度花在 PDF 解析上，专利版完全不需要——这就是"数据源质量决定工程复杂度"的直观体现。

### 检索效果差异

| 维度 | 年报版 | 专利版 |
|------|-------|--------|
| BM25 的价值 | 精确匹配数字（"1476.94亿元"） | 精确匹配术语（"Doppler Velocity Log"、"thruster"） |
| 跨文档查询 | 同公司多年对比（"茅台2021-2023营收趋势"） | 不同公司技术路线对比（"DJI vs Kongsberg 推进方案"） |
| 特有 chunk 类型 | 表格（财务报表） | 权利要求（claim，法律主张） |
| 数据量 | 10,353 chunks | 233 chunks（有意精简，方便教学） |

---

## 评测题集

测试题集位于 `evaluation/questions.json`，共 20 题，覆盖 5 类：

| 类型 | 题数 | 考察能力 | 示例 |
|------|------|---------|------|
| `simple_fact` | 5 | 基础单专利检索 | AUV 最常组合使用的传感器是什么？ |
| `precise_number` | 5 | BM25 术语召回 | 水下SLAM中声学vs视觉的误差来源？ |
| `cross_doc_compare` | 5 | 跨专利技术路线对比 | 螺旋桨推进 vs 仿生推进的优劣？ |
| `time_trend` | 2 | 技术发展趋势分析 | 深度学习在水下机器人的应用趋势？ |
| `should_refuse` | 3 | 幻觉控制/拒答 | 某公司产品售价？军事装备参数？ |

---

## 踩坑与注意事项

### 1. DASHSCOPE_API_KEY 在新终端中读不到

**原因**：通过 Windows 系统属性对话框设置的环境变量，只对新启动的进程生效。已打开的终端/IDE 不会自动刷新。

**解决**：
```powershell
# 方法一：关闭当前终端，重新打开一个新终端
# 方法二：在当前终端手动刷新
$env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
```

### 2. parse_patents.py 只输出了 1 个 block

**原因**：`main()` 函数从 `manifest.json` 读取数据传给 `parse_patent()`，但 manifest 只有元数据（patent_id、title、assignee 等），没有 `abstract`、`description`、`claims` 字段。

**解决**：修改 `main()` 从 `data/raw/*.json` 加载完整专利数据，已修复。

### 3. Google Patents API 返回 503 / 搜索无结果

**原因**：Google Patents 搜索页面依赖 JavaScript 渲染，`requests` 无法直接抓取；详情页面对爬虫请求返回 503。

**解决**：放弃 API 抓取方案，改为内置高质量专利数据集。对教学项目而言，内置数据比依赖外部 API 更稳定可靠。

### 4. PatentsView API 从国内无法访问

旧版 API（`api.patentsview.org`）已迁移至新平台，新平台返回 Angular SPA 页面而非 JSON。国内还存在 SSL 连接问题。

### 5. text-embedding-v3 批次上限

DashScope text-embedding-v3 单次最多 10 条（不是文档中的 25）。`BATCH_SIZE = 10`，已硬编码在 `build_index.py` 中。

### 6. BM25 阈值不能复用 vec_score

RRF 融合后的 `rrf_score` 量纲与余弦相似度完全不同（~0.016 vs ~0.6），相关性阈值检查只使用 `vec_score`。这在 `rag_pipeline.py` 中已正确处理。

---

## 目录结构

```
week10_patent_rag/
├── readme.md                     ← 本文档
├── ARCHITECTURE.md               ← 详细技术架构说明
├── requirements.txt              ← Python 依赖
│
├── src/                          ← 核心代码（原生实现，无 LangChain 版）
│   ├── fetch_patents.py          # 专利数据获取/生成
│   ├── parse_patents.py          # JSON → 结构化 blocks
│   ├── chunk_documents.py        # blocks → chunks（3 种策略）
│   ├── build_index.py            # chunks → FAISS 向量索引
│   ├── rag_pipeline.py           # 问答流水线（BM25+向量+RRF+Rerank+LLM）
│   ├── serve.py                  # FastAPI HTTP 服务
│   └── static/index.html         # 教学可视化（深海蓝主题）
│
├── data/                         # 数据中间产物
│   ├── manifest.json             # 专利元数据索引
│   ├── raw/                      # 21 份专利 JSON（完整文本）
│   ├── parsed/                   # 21 份解析结果（266 blocks）
│   └── chunks/                   # 分块结果（all_semantic.json，233 chunks）
│
├── vectorstore/                  # 向量库
│   ├── faiss_index.bin           # FAISS 索引（932 KB，233×1024维）
│   └── faiss_meta.json           # 元数据（含每条 chunk 完整文本）
│
└── evaluation/                   # 评估
    └── questions.json            # 20 道标准评测题
```

---

## 扩展方向

1. **Graph RAG**：专利引用关系（前引/后引）是天然的文档图，可作为第二阶段教学内容
2. **多语言检索**：中英文专利混合检索 + 跨语言 embedding 质量评估
3. **技术路线图分析**：利用 LLM 推理能力，从多份专利中提取技术演进脉络
4. **增加数据量**：将内置数据集的 21 份扩展到 100+ 份，测试 RRF 融合在大规模语料上的表现
5. **权利要求对比**：利用 claim 独立分块的优势，做"侵权风险分析"式的高精度检索
6. **对接真实 API**：当国内网络条件改善后，对接 PatentsView 或 EPO OPS 实现动态数据获取
