# 技术架构说明

> 水下机器人专利 RAG 问答系统 — 整体方案、选型决策与设计原理

---

## 一、项目定位

本项目以"水下机器人专利技术智能问答"为场景，构建一套 RAG（检索增强生成）系统。数据来自 Google Patents 公开专利数据库，覆盖 AUV/ROV/水下无人机/水下通信/海底测绘等领域的中英文专利。

**与 week10_rag（上市公司年报）的核心对比**：

| 维度 | week10 年报版 | 本专利版 |
|------|-------------|---------|
| 数据源 | 巨潮资讯网 PDF（~85MB） | Google Patents XML/JSON |
| 解析复杂度 | 高（pdfplumber + PyMuPDF + OCR，~300行） | 低（JSON 字段映射，~100行） |
| 分块核心差异 | 财务表格单独成块 | 权利要求逐条独立成块 |
| 检索特点 | 数字精确匹配（BM25 优势） | 技术术语精确匹配（BM25 优势更显著） |
| 跨文档挑战 | 同公司多年数据对比 | 不同专利权人的技术路线对比 |
| 技术栈 | DashScope API（向量 + LLM） | 完全相同（验证了管道的通用性） |

**设计理念**：两个项目共用同一套代码架构（数据流完全一致），但数据源从"脏 PDF"变为"干净 XML"，让学生直面"数据源质量决定工程复杂度"这一核心工程认知。

---

## 二、整体流水线

```
Google Patents API
      │
      ▼ fetch_patents.py
专利原始 JSON（标题 + 摘要 + 权利要求 + 说明书全文）
      │
      ▼ parse_patents.py
结构化 blocks（title / abstract / claim / text + 章节路径）
      │
      ▼ chunk_documents.py
文档分块（三种策略可切换，语义分块新增 claim 类型处理）
      │
      ▼ build_index.py
向量化 + FAISS 索引（DashScope text-embedding-v3，1024 维）
      │
      ▼ rag_pipeline.py
问答流水线：查询 → 向量检索 + BM25 → RRF 融合 → (Rerank) → LLM 生成
      │
      ▼ evaluation/
评估（20 道评测题）
```

---

## 三、各环节技术说明

### 3.1 数据获取（fetch_patents.py）

**Google Patents**（patents.google.com）是全球专利检索平台，覆盖 CN/US/EP/WO/JP 等主要专利局。

**获取策略**：
1. Phase 1：手工精选的种子专利列表（经典/代表性水下机器人专利），确保高质量收录
2. Phase 2：15+ 组中英文关键词搜索（覆盖整体 + 子技术领域 + 公司），批量发现
3. 对每条结果，从页面中提取 JSON-LD 结构化数据（标题/摘要/权利要求/说明书/专利权人/发明人）

**与年报版的关键差异**：
- 年报通过巨潮 API 的 `POST` 表单查询；专利通过 Google Patents 公开页面解析
- 年报输出 PDF 二进制文件；专利输出 JSON 文本文件
- 数据获取从"下载大文件"变为"解析结构化文本"，速度更快、存储更小

### 3.2 专利文本解析（parse_patents.py）

**数据结构**：与 week10 完全兼容的 `ParsedBlock`：
```python
@dataclass
class ParsedBlock:
    block_type:   str          # "title" | "abstract" | "claim" | "text"
    content:      str
    page_num:     int = 0      # 专利无页码概念
    section_path: list[str]    # ["专利标题", "说明书", "具体实施方式"]
    claim_num:    int = 0      # 权利要求编号
```

**解析逻辑**：
- title → 独立块
- abstract → 独立块（高质量检索入口）
- description → 按章节标题（背景技术/发明内容/具体实施方式）拆分 → 段落块
- claims → 按编号（"1."、"2."）逐条拆分 → 每条独立块

**与年报版的关键对比**：
- 年报版用了 3 个库（pdfplumber、PyMuPDF、Tesseract）处理 PDF 噪声
- 专利版只需 `json.load()` + 字符串拆分，代码量约 1/3
- 这正是"数据源质量决定工程复杂度"的直观教材

### 3.3 文档分块（chunk_documents.py）

与 week10 相同的三种策略，新增对专利特有块类型的处理：

| 策略 | 说明 | 专利版调整 |
|------|------|-----------|
| `fixed` | 500 字符固定切 | 无变化 |
| `semantic`（默认） | 标题/摘要/权利要求独立成块，说明书段落累积到 800 字 | 新增 `claim` 和 `abstract` 类型处理 |
| `hierarchical` | 父子块（2000/400 字符） | 无变化 |

**权利要求（claim）的分块逻辑**：每条 claim 是一个独立的法律保护范围声明，必须单独成块。不能按字符数硬切——切断一条 claim 等于破坏其法律含义。

### 3.4 Embedding + 向量库

与 week10 完全一致：
- **Embedding**：DashScope text-embedding-v3，1024 维，批次上限 10 条
- **向量库**：FAISS IndexFlatIP（内积 = 余弦相似度）
- **元数据字段**：patent_id / title / assignee / patent_office / section / block_types / claim_num

### 3.5 检索策略

与 week10 完全一致的混合检索架构：
- **向量检索**：FAISS 余弦相似度
- **BM25**：jieba 分词 + BM25Okapi
- **RRF 融合**：k=60
- **可选 Rerank**：BAAI/bge-reranker-base

专利场景下 BM25 的优势比年报更显著——技术术语（"Doppler velocity log"、"thruster"、"hydrophone"）的精确匹配需求远高于财务数字。

### 3.6 LLM 生成

- **模型**：DashScope qwen-plus（同 week10）
- **System Prompt**：专利技术分析师角色
- **核心约束**：只从专利资料回答 + 标注来源编号 + 区分独立/从属权利要求

---

## 四、评估体系

### 4.1 评测题集（20 题）

| 类型 | 题数 | 考察能力 | 示例 |
|------|------|---------|------|
| `simple_fact` | 5 | 基础单专利检索 | AUV 最常组合使用的传感器是什么？ |
| `precise_number` | 5 | BM25 术语召回 | 水下SLAM中声学vs视觉的误差来源？ |
| `cross_doc_compare` | 5 | 跨专利技术路线对比 | 螺旋桨推进 vs 仿生推进的优劣？ |
| `time_trend` | 2 | 技术发展趋势分析 | 深度学习在水下机器人的应用趋势？ |
| `should_refuse` | 3 | 幻觉控制/拒答 | 某公司产品售价？军事装备参数？ |

### 4.2 与年报版评估题的结构对应

两个项目的评估题集采用相同的 5 类分组和编号结构（1-20），便于交叉对比：
- 年报版考察"财务数字精确提取"→ 专利版考察"技术术语精确提取"
- 年报版跨文档对比"茅台 vs 五粮液"→ 专利版跨专利对比"DJI vs Kongsberg"
- 年报版拒答题"股价/投资建议"→ 专利版拒答题"产品售价/军事信息"

---

## 五、目录结构

```
week10_patent_rag/
├── src/
│   ├── fetch_patents.py         # Google Patents → data/raw/
│   ├── parse_patents.py         # JSON → 结构化 blocks
│   ├── chunk_documents.py       # blocks → chunks（3 种策略）
│   ├── build_index.py           # chunks → FAISS 索引
│   ├── rag_pipeline.py          # 问答流水线（BM25+向量+RRF+Rerank+LLM）
│   ├── serve.py                 # FastAPI HTTP 服务
│   └── static/index.html        # 教学可视化（深海蓝主题）
├── data/
│   ├── raw/                     # 专利 JSON 原始数据
│   ├── manifest.json
│   ├── parsed/                  # 解析后的 blocks
│   └── chunks/                  # 分块结果
├── vectorstore/
│   ├── faiss_index.bin
│   └── faiss_meta.json
├── evaluation/
│   └── questions.json           # 20 道评测题
├── requirements.txt
└── ARCHITECTURE.md
```

---

## 六、与 week10_rag 的代码复用度

```
fetch_patents.py     ██████████ 全新  （数据源从巨潮→Google Patents）
parse_patents.py     ██████████ 全新  （解析从 PDF→JSON，代码量反而是 1/3）
chunk_documents.py   ▏          5%改动 （新增 claim/abstract 类型）
build_index.py       ▏          5%改动 （元数据字段名适配）
rag_pipeline.py      ▏          5%改动 （System prompt 适配）
serve.py             ▏          5%改动 （路径、文案）
index.html           ▎          改皮   （深海蓝主题）
```

关键在于：**从分块到生成的核心管道（chunk → build → pipeline → serve）几乎原封不动**，验证了这套 RAG 架构的领域可迁移性。

---

## 七、扩展方向

1. **Graph RAG**：专利引用关系（前引/后引）是天然的文档图，可作为第二阶段教学内容
2. **多语言检索**：中英文专利混合检索 + 跨语言 embedding（text-embedding-v3 已支持多语言）
3. **技术路线图分析**：利用大模型的推理能力，从多份专利中提取技术演进脉络
4. **专利侵权预警**：给定一份技术方案描述，检索最相似的已授权专利（高精度召回场景）
