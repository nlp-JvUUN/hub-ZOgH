# RAG 年报问答系统 — 流水线详解

本文档详细介绍本项目的**离线数据构建**和**在线检索问答**两条核心流程，结合代码说明每一步的设计原理和关键技术细节。

---

## 目录

- [一、项目结构概览](#一项目结构概览)
- [二、离线数据构建流程](#二离线数据构建流程)
  - [步骤 1：文档加载与解析](#步骤-1文档加载与解析)
  - [步骤 2：文档分块](#步骤-2文档分块)
  - [步骤 3：向量化与索引构建](#步骤-3向量化与索引构建)
- [三、在线检索流程](#三在线检索流程)
  - [步骤 ①：查询改写（可选）](#步骤-查询改写可选)
  - [步骤 ②：向量检索](#步骤-向量检索)
  - [步骤 ③：BM25 关键词检索](#步骤-bm25-关键词检索)
  - [步骤 ④：RRF 融合排名](#步骤-rrf-融合排名)
  - [步骤 ⑤：CrossEncoder 重排序（可选）](#步骤-crossencoder-重排序可选)
  - [步骤 ⑥：相关性阈值过滤](#步骤-相关性阈值过滤)
  - [步骤 ⑦：LLM 生成](#步骤-llm-生成)
- [四、两条流程的关系全景图](#四两条流程的关系全景图)
- [五、关键设计决策](#五关键设计决策)

---

## 一、项目结构概览

```
rag_annual_report/
├── src/                              # 原生版本（企业级）
│   ├── download_reports.py           # 从巨潮资讯网下载 PDF
│   ├── parse_pdf.py                  # PDF 解析（pdfplumber + PyMuPDF + OCR）
│   ├── parse_documents.py            # 多格式文档解析——统一入口
│   ├── parsed_block_schema.py        # 共享数据模型（ParsedBlock / ParsedDocument）
│   ├── format_loaders/               # 格式加载器（每种文件格式一个 loader）
│   │   ├── __init__.py               # 加载器注册表 + 调度函数
│   │   ├── base_loader.py            # Protocol 接口 + 共享工具函数
│   │   ├── pdf_loader.py             # PDF 加载器
│   │   ├── txt_loader.py             # TXT 加载器
│   │   ├── markdown_loader.py        # Markdown 加载器
│   │   ├── csv_loader.py             # CSV 加载器
│   │   ├── docx_loader.py            # Word 加载器
│   │   ├── html_loader.py            # HTML 加载器
│   │   ├── xlsx_loader.py            # Excel 加载器
│   │   └── image_loader.py           # 图片 OCR 加载器
│   ├── chunk_documents.py            # 三种分块策略（固定/语义/层级）
│   ├── build_index.py                # FAISS 向量索引 + DashScope embedding
│   ├── rag_pipeline.py               # 完整问答流水线
│   ├── serve.py                      # FastAPI HTTP 服务
│   └── static/index.html             # 可视化 Web 页面
├── src_langchain/                    # LangChain 版本（框架原型）
├── evaluation/                       # 评估体系（RAGAS + 消融实验）
├── data/
│   ├── raw/                          # 多格式原始文档
│   ├── raw_pdf/                      # PDF 原始文件（向后兼容）
│   ├── parsed/                       # 解析后的统一 JSON
│   └── chunks/                       # 分块后的 JSON
├── vectorstore/                      # FAISS 向量索引
├── models/                           # 本地模型（BGE 等）
└── requirements.txt
```

---

## 二、离线数据构建流程

离线流程是一条"手动/定时执行"的流水线，把原始文档变成可检索的向量索引。

### 整体链路

```
原始文档 → 解析(parse) → 分块(chunk) → 向量化+建索引(build_index)
   ↓            ↓              ↓                    ↓
data/raw/   data/parsed/   data/chunks/       vectorstore/
(各种格式)    (统一JSON)     (分块JSON)         (FAISS索引)
```

---

### 步骤 1：文档加载与解析

**入口文件**：[src/parse_documents.py](src/parse_documents.py)

核心逻辑：

1. 扫描 `data/raw/` 和 `data/raw_pdf/` 目录
2. 根据文件扩展名 → 从注册表获取对应 loader
3. `loader.load()` → 输出统一的 `ParsedDocument`
4. 保存为 `data/parsed/{stem}.json`

#### 格式分发机制

**文件**：[src/format_loaders/__init__.py](src/format_loaders/__init__.py)

```python
EXTENSION_MAP: dict[str, str] = {
    ".pdf":  "PdfLoader",
    ".txt":  "TxtLoader",
    ".docx": "DocxLoader",
    ".doc":  "DocxLoader",
    ".md":   "MarkdownLoader",
    ".html": "HtmlLoader",
    ".htm":  "HtmlLoader",
    ".csv":  "CsvLoader",
    ".xlsx": "XlsxLoader",
    ".xls":  "XlsxLoader",
    ".png":  "ImageLoader",
    ".jpg":  "ImageLoader",
    ".jpeg": "ImageLoader",
    ".bmp":  "ImageLoader",
    ".tiff": "ImageLoader",
    ".tif":  "ImageLoader",
}
# 共 16 种扩展名映射到 8 个加载器
```

`get_loader(file_path)` 通过**懒加载**按需导入——解析 TXT 时不需要 pdfplumber，解析 PDF 时不需要 python-docx。只有真正需要创建某个 loader 实例时才会导入对应模块。

#### 统一的输出格式

**文件**：[src/parsed_block_schema.py](src/parsed_block_schema.py)

这是整个系统的**核心合约**——所有加载器都输出这个格式，所有下游消费方都读取这个格式。

```python
@dataclass
class ParsedBlock:
    """
    一个解析块 = 文档里的一段连续内容（文字段落 / 表格 / 标题）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户来源位置。
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int            # 页码（无分页的格式用 0）
    section_path: list[str]      # 章节路径，如 ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能较低
    raw_table:    Optional[list] = None   # 原始表格数据


@dataclass
class ParsedDocument:
    """
    一个解析后的文档 = 元信息 + 源文件路径 + 解析块列表
    """
    meta:   dict              # {"stock_code": "600519", "year": "2023", "company_name": "贵州茅台", ...}
    source: str               # 源文件绝对路径
    blocks: list[ParsedBlock]
```

保存为 JSON 后的结构示例：

```json
{
  "meta": {
    "stock_code": "600519",
    "year": "2023",
    "company_name": "贵州茅台",
    "filename": "600519_2023_贵州茅台_2023年年度报告.pdf"
  },
  "source": "/path/to/600519_2023_贵州茅台_2023年年度报告.pdf",
  "blocks": [
    {
      "block_type": "title",
      "content": "第三章 管理层讨论与分析",
      "page_num": 21,
      "section_path": ["第三章 管理层讨论与分析"],
      "is_ocr": false,
      "raw_table": null
    },
    {
      "block_type": "text",
      "content": "报告期内，公司实现营业收入1476亿元...",
      "page_num": 22,
      "section_path": ["第三章 管理层讨论与分析", "一、经营情况概述"],
      "is_ocr": false,
      "raw_table": null
    },
    {
      "block_type": "table",
      "content": "| 项目 | 金额 | 增长率 |\n| --- | --- | --- |\n| 营业收入 | 1476亿 | 19% |",
      "page_num": 38,
      "section_path": ["第三章 管理层讨论与分析", "一、经营情况概述"],
      "is_ocr": false,
      "raw_table": [["项目", "金额", "增长率"], ["营业收入", "1476亿", "19%"]]
    }
  ]
}
```

#### 各格式加载器的处理策略

| 格式 | 标题检测 | 表格检测 | 文字提取 | page_num | 依赖 |
|------|---------|---------|---------|----------|------|
| **PDF** | 字号≥14pt + 加粗 + 章节正则 | pdfplumber 表格算法 | PyMuPDF `get_text("dict")` + 扫描页 OCR | 实际页码 | pdfplumber, pymupdf, pytesseract |
| **TXT** | 章节正则（第一章、一、1. 等） | 不支持 | 按 `\n\n` 分隔段落 | 0 | 无（标准库） |
| **Markdown** | `#` 标题标记（1-6级） | `\|...\|` 管道表格 | 段落文本 + 代码块 | 0 | 无（标准库 re） |
| **DOCX** | Heading 样式 + 加粗短文本 | python-docx 原生表格 API | 段落文本 | 0 | python-docx |
| **HTML** | `<h1>`-`<h6>` 标签 | `<table>` 元素 | `<p>`, `<div>`, `<li>` 等 | 0 | beautifulsoup4 |
| **CSV** | 不支持 | 整个文件为一张表 | 不支持（纯表格） | 0 | 无（标准库 csv） |
| **XLSX** | 不支持 | 每个 Sheet 为一张表 | 不支持（纯表格） | sheet 序号 | openpyxl |
| **图片** | 不支持 | 不支持 | OCR 全文识别 | 0 | Pillow, pytesseract |

#### 编码检测

**文件**：[src/format_loaders/base_loader.py](src/format_loaders/base_loader.py) — `detect_encoding()`

对于 TXT/CSV 等文本文件，需要自动检测编码：

```python
def detect_encoding(file_path: Path) -> str:
    # 优先使用 chardet（更准确）
    try:
        import chardet
        result = chardet.detect(raw)
        if result and result.get("encoding"):
            return normalize_encoding(result["encoding"])
    except ImportError:
        pass

    # 简单 try-chain：UTF-8 → GBK → GB2312 → latin-1
    for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"
```

这对于中文年报文档很重要——许多 Windows 导出的文件使用 GBK/GB2312 编码。

#### 章节层级维护

所有加载器共享同一个 `_update_section()` 逻辑，维护一个章节栈：

```python
def _update_section(title, stack):
    """根据标题模式推断章节层级"""
    if re.match(r"^第[一二三四五六七八九十]+章", title):
        stack.clear(); stack.append(title)        # 顶级章
    elif re.match(r"^第[一二三四五六七八九十]+节", title):
        stack[:] = stack[:1] + [title]            # 二级节
    elif re.match(r"^[一二三四五六七八九十]、", title):
        stack[:] = stack[:2] + [title]            # 三级
    else:
        stack.append(title)                        # 追加
```

---

### 步骤 2：文档分块

**入口文件**：[src/chunk_documents.py](src/chunk_documents.py)

读取 `data/parsed/*.json`，将 blocks 切分成适合 embedding 的 chunk。三种策略：

#### 策略 A — Fixed（固定大小）

```python
def chunk_fixed(text, chunk_size=500, overlap=50):
    start = 0
    while start < len(text):
        yield text[start:start + chunk_size]
        start += chunk_size - overlap
```

- **优点**：实现最简单，块大小可预测
- **缺点**：无视句子/段落边界，表格会被切断

#### 策略 B — Semantic（语义分块，默认推荐）

```python
for block in blocks:
    if block["block_type"] == "title":
        flush(buffer)           # 遇到标题 → 先刷新缓冲区
        yield title_as_chunk    # 标题单独成块
    elif block["block_type"] == "table":
        flush(buffer)           # 表格之前的内容
        yield table_as_chunk    # 表格单独成块（不与文字混合）
    else:  # text
        buffer.append(block)    # 文字累积到 max_chunk_size=800 再切
```

这是一个**结构感知**的分块器：

- **标题强制切块**：保持章节边界完整，每个标题独立成块
- **表格独立成块**：防止表格与文字混合导致 embedding 质量下降
- **文字段落尽量合并**：最大 800 字符，保留语义完整性

#### 策略 C — Hierarchical（层级分块）

```
父块 (2000字符) ← 给 LLM 看，信息完整
  ├── 子块 (400字符) ← 向量检索用，精度高
  ├── 子块 (400字符)
  └── 子块 (400字符)
```

每个子块携带 `parent_id` 和 `parent_content`：

```json
{
  "chunk_id": "600519_2023_00042",
  "content": "...400字...",       // 子块内容（向量检索）
  "metadata": {
    "parent_id": "a1b2c3d4",
    "parent_content": "...2000字..."  // 父块内容（给 LLM 读）
  }
}
```

检索时：命中子块 → 取 `parent_id` → 给 LLM 读父块的完整内容。这就是 **"Small-to-Big Retrieval"**。

#### 输出格式

`data/chunks/all_{strategy}.json`，每个 chunk：

```json
{
  "chunk_id": "600519_2023_00001",
  "content": "2023年公司实现营业收入1476亿元，同比增长19%...",
  "metadata": {
    "stock_code": "600519",
    "year": "2023",
    "page_num": 38,
    "section": "第三章 > 一、经营情况概述",
    "block_types": ["text"],
    "is_ocr": false,
    "strategy": "semantic",
    "source_file": "600519_2023_xxx.json"
  }
}
```

---

### 步骤 3：向量化与索引构建

**入口文件**：[src/build_index.py](src/build_index.py)

这是离线流程的最后一步，也是最关键的一步。

```python
# 1. 读取所有 chunks
with open("data/chunks/all_semantic.json") as f:
    chunks = json.load(f)   # 例如 10353 个 chunk

# 2. 提取文本
texts = [c["content"] for c in chunks]

# 3. 批量调用 DashScope API 计算 embedding
embeddings = embed_texts(client, texts)
# 返回 shape=(10353, 1024) 的 float32 数组

# 4. L2 归一化（使内积等价于余弦相似度）
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 5. 构建 FAISS 索引
index = faiss.IndexFlatIP(1024)   # IP = Inner Product（内积）
index.add(embeddings)

# 6. 持久化
faiss.write_index(index, "vectorstore/faiss_semantic/faiss_index.bin")
json.dump(meta_list, "vectorstore/faiss_semantic/faiss_meta.json")
```

#### 关键技术细节

**Embedding 批处理**：

```python
BATCH_SIZE = 10             # DashScope text-embedding-v3 单次最多 10 条
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024            # 可选 768 / 512 节省存储
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**带重试的 API 调用**：

```python
def embed_texts(client, texts):
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(3):          # 失败重试 3 次
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
                )
                vecs = [e.embedding for e in resp.data]
                break
            except Exception as e:
                if attempt == 2: raise     # 最后一次失败直接抛出
                time.sleep(2 ** attempt)   # 指数退避：1s, 2s, 4s
```

**L2 归一化**：

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-9)   # 防止除零
embeddings = embeddings / norms
```

归一化后，`IndexFlatIP`（内积检索）等价于余弦相似度检索，分数范围 0~1，物理意义明确。

**FAISS 索引选择**：

| 索引类型 | 适用场景 | 本项目选择原因 |
|----------|---------|--------------|
| `IndexFlatIP` | 数据量 < 10万 | 精确暴力检索，无需训练，速度够快 |
| `IndexIVFFlat` | 10万~1000万 | 需要聚类训练，有精度损失 |
| `IndexHNSW` | 100万+ | 图索引，内存占用大 |
| `IndexPQ` | 内存受限 | 有损压缩，精度下降 |

本项目 ~1万 chunk，`IndexFlatIP` 完全够用。

**元数据分离存储**：

```python
# 索引文件：纯向量（faiss_index.bin, ~41MB）
index_path = VECTORSTORE_DIR / "faiss_index.bin"
faiss.write_index(index, str(index_path))

# 元数据：文本 + 溯源信息（faiss_meta.json, ~15MB）
meta_path = VECTORSTORE_DIR / "faiss_meta.json"
json.dump(meta_list, meta_path)
```

两者通过 FAISS 返回的数组索引（`idx`）关联：
- `faiss_index.bin[i]` = 第 i 个向量的 1024 维数据
- `faiss_meta.json[i]` = 第 i 个向量的文本和元信息

分开存储的好处：向量数据保持高效二进制格式，元数据可读可编辑，互不干扰。

---

## 三、在线检索流程

在线流程是"每次用户提问都执行"的实时流水线。

### 整体链路

```
用户问题 → 查询改写 → 向量检索 + BM25检索 → RRF融合 → Rerank → 阈值过滤 → LLM生成 → 答案+引用
   ↓         ↓          ↓            ↓        ↓        ↓         ↓          ↓         ↓
  string   qwen-turbo   FAISS       jieba   加权排名  BGE模型   余弦>0.25  qwen3.7   标注来源
```

所有这些都封装在 `RAGPipeline` 类中 — [src/rag_pipeline.py:303-367](src/rag_pipeline.py#L303-L367)。

---

### 步骤 ①：查询改写（可选）

**文件**：[src/rag_pipeline.py:228-254](src/rag_pipeline.py#L228-L254)

```python
def rewrite_query(query, client):
    """用 qwen-turbo 将模糊问题改写为精确检索查询"""
    resp = client.chat.completions.create(
        model="qwen-turbo",             # 最快最便宜，够用
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索查询优化专家。将用户的问题改写为"
                    "更适合从年度报告中检索信息的精确查询语句。"
                    "保留关键实体（公司名、年份、财务指标），"
                    "扩展相关关键词，不要超过50字。"
                    "直接输出改写后的查询语句，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
```

**效果示例**：

| 原始问题 | 改写后 |
|---------|--------|
| "茅台最近怎么样" | "贵州茅台2023年营业收入净利润同比增长率经营情况" |
| "五粮液赚钱能力" | "五粮液2023年净利润净利率毛利率盈利能力" |
| "宁德时代研发投入" | "宁德时代2023年研发费用研发投入占营业收入比例" |

改写增加了同义词和相关指标，提高了检索召回率。使用 `qwen-turbo` 而非 `qwen3.7-plus` 是因为改写不需要高质量推理，最快最便宜的模型即可。

---

### 步骤 ②：向量检索（语义召回）

**文件**：[src/rag_pipeline.py:79-124](src/rag_pipeline.py#L79-L124)

```python
class VectorStore:
    def __init__(self, client):
        import faiss
        self.client    = client
        self.index     = faiss.read_index(str(INDEX_PATH))     # 加载向量索引
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)                      # 加载元数据

    def search(self, query, top_k=10, filter_meta=None):
        # 1. 查询向量化
        resp = self.client.embeddings.create(
            model="text-embedding-v3", input=[query], dimensions=1024
        )
        query_vec = np.array([resp.data[0].embedding], dtype="float32")
        query_vec = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-9)  # L2 归一化

        # 2. FAISS 内积搜索（等价于余弦相似度）
        scores, indices = self.index.search(query_vec, top_k * 4)
        # 多取 4 倍——元数据过滤可能丢弃一些结果

        # 3. 按分数 + 可选元数据过滤
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])      # 从 meta JSON 取回文本和溯源信息
            item["vec_score"] = float(score)       # 余弦相似度分数

            # 元数据过滤：如 {"stock_code": "600519", "year": "2023"}
            if filter_meta:
                if not all(str(item.get(k, "")) == str(v)
                           for k, v in filter_meta.items()):
                    continue  # 不匹配则跳过

            results.append(item)
            if len(results) >= top_k:
                break
        return results
```

**检索流程图**：

```
user query: "茅台2023年营收"
        │
        ▼
  DashScope API (1次调用)
        │
        ▼
  query_vec: [0.023, -0.015, ..., 0.041]  (1024维, L2归一化)
        │
        ▼
  FAISS IndexFlatIP.search(query_vec, k=40)
        │  < 1ms, 暴力内积扫描 10353 个向量
        ▼
  scores: [0.892, 0.876, 0.854, ...]
  indices: [4521, 2308, 671, ...]
        │
        ▼
  meta_list[indices[i]] → 文本 + 元数据
        │
        ▼
  filter by stock_code / year (if specified)
        │
        ▼
  return top_k results with vec_score
```

**为什么 `search(top_k * 4)`**：元数据过滤会丢弃不匹配的结果，所以先从向量库中多取一些候选，过滤后再截断到 `top_k`。

---

### 步骤 ③：BM25 关键词检索（精确召回）

**文件**：[src/rag_pipeline.py:129-160](src/rag_pipeline.py#L129-L160)

```python
class BM25Store:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        # 分词整个语料库（启动时做一次，约数秒）
        tokenized = [list(jieba.cut(item["content"]))
                     for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=10):
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)      # 对每个文档打分
        top_idx = np.argsort(scores)[::-1][:top_k]  # 取 top-k

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:   # 完全不相关的跳过
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results
```

**BM25 原理简述**：

BM25 是经典的词频-逆文档频率（TF-IDF）算法的改进版。核心思想：

- 一个词在当前文档中出现次数越多，得分越高（**TF 效应**，但有饱和上限）
- 一个词在语料库中出现的文档数越少，得分越高（**IDF 效应**）
- 文档长度越长，TF 得分会被适当惩罚（**长度归一化**）

公式：`score(d, q) = Σ IDF(qi) × TF(qi, d) / (k1 × (1 - b + b × |d| / avgdl) + TF(qi, d))`

**为什么需要 BM25 + 向量检索双路**：

| 场景 | 向量检索 | BM25 |
|------|---------|------|
| "公司经营情况如何"（语义查询） | ✅ 好 | ❌ 差 |
| "1476亿元"（精确数字） | ❌ 差 | ✅ 好 |
| "600519"（股票代码） | ❌ 差 | ✅ 好 |
| "研发费用资本化率"（专业术语） | ⚠️ 一般 | ✅ 好 |

两路互补——语义检索处理模糊概念，关键词检索处理精确匹配。

---

### 步骤 ④：RRF 融合排名

**文件**：[src/rag_pipeline.py:165-194](src/rag_pipeline.py#L165-L194)

```python
def reciprocal_rank_fusion(vec_results, bm25_results, k=60):
    """
    RRF 公式：score(d) = Σ 1/(k + rank_i(d))

    k=60 是经验值：
      让排名靠前的得分优势不明显衰减
      排名 1 的分数 = 1/61 ≈ 0.0164
      排名 10 的分数 = 1/70 ≈ 0.0143
      差距只有 ~15%，防止某一路过于主导
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    # 向量检索的排名贡献
    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    # BM25 检索的排名贡献
    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    # 按 RRF 分数降序排列
    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    return [dict(chunk_map[cid]) for cid in sorted_cids]
```

**RRF 融合示意**：

```
                   向量检索排名          BM25 检索排名
Chunk A (最佳匹配)     1  → 1/61=0.0164      3  → 1/63=0.0159   总分: 0.0323 ✅ 排第1
Chunk B              5  → 1/65=0.0154      1  → 1/61=0.0164   总分: 0.0318 ✅ 排第2
Chunk C              2  → 1/62=0.0161      8  → 1/68=0.0147   总分: 0.0308
Chunk D             10  → 1/70=0.0143      2  → 1/62=0.0161   总分: 0.0304
```

**为什么用 RRF 而不是加权求和**：

- 不同检索方式的分数量纲不同：
  - 向量检索：余弦相似度 0~1
  - BM25：可以是 0~50+
- 直接加权需要归一化，而各检索器的分数量纲和分布都不同
- RRF 只关心排名，天然跨检索器可比，无需归一化
- 无需训练任何权重参数

---

### 步骤 ⑤：CrossEncoder 重排序（可选）

**文件**：[src/rag_pipeline.py:198-223](src/rag_pipeline.py#L198-L223)

```python
def rerank(query, candidates, top_k=4):
    """用 BGE-Reranker 对候选集二次精排"""
    try:
        from sentence_transformers import CrossEncoder

        # 本地模型路径优先，否则从 HuggingFace 下载
        model_path = "models/bge-reranker-base"
        model_name = str(model_path) if Path(model_path).exists() \
                     else "BAAI/bge-reranker-base"
        reranker = CrossEncoder(model_name)

        # 对每个 (query, chunk) 对打分
        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(key=lambda x: -x.get("rerank_score", 0))
    except ImportError:
        logger.warning("sentence-transformers 未安装，跳过 Rerank")
    except Exception as e:
        logger.warning(f"Rerank 失败，使用 RRF 原始排序: {e}")

    return candidates[:top_k]
```

**为什么需要 Rerank（重排序）**：

| 阶段 | 模型 | 类型 | 特点 |
|------|------|------|------|
| 向量检索（粗筛） | text-embedding-v3 | **Bi-Encoder** | query 和 doc 独立编码，速度快，可预先计算所有 doc 向量；但 query-doc 交互不充分，精度有限 |
| Rerank（精排） | bge-reranker-base | **CrossEncoder** | query 和 doc 联合输入，双向注意力，精度高；但不能预先计算，只能对小候选集打分 |

**组合拳**：Bi-Encoder 从 10353 个候选中粗筛 10 个（快）→ CrossEncoder 精排到 4 个（准）。

失败时自动降级：如果 `sentence-transformers` 未安装或模型加载失败，直接取 RRF 排序的 top-k，不影响主流程。

---

### 步骤 ⑥：相关性阈值过滤

**文件**：[src/rag_pipeline.py:349-361](src/rag_pipeline.py#L349-L361)

```python
# 阈值始终用 vec_score（余弦相似度，0~1 可解释）
top_score = final[0].get("vec_score", final[0].get("rerank_score", 1.0))
if top_score < SCORE_THRESHOLD and filter_meta is None:
    return {
        "answer": "根据年报知识库未能找到与该问题相关的内容，建议直接查阅原始年报。",
        "citations": [], "retrieved": final,
    }
```

**设计要点**：

- `SCORE_THRESHOLD = 0.25`：当最高余弦相似度低于 0.25 时，认为检索结果不可靠
- 阈值检查**始终用 `vec_score`**（而非 `rrf_score`）：
  - `vec_score` = 归一化余弦相似度，范围 0~1，物理意义明确
  - `rrf_score` = 排名倒数之和，量纲约 0.016，无法直接与阈值比较
- 当用户使用了过滤器（指定股票代码/年份）时**跳过阈值检查**：
  - 用户主动缩小范围说明其对数据有预期
  - 即使 cosine similarity 偏低，也应该尝试回答

---

### 步骤 ⑦：LLM 生成

**文件**：[src/rag_pipeline.py:259-298](src/rag_pipeline.py#L259-L298)

#### 上下文组装

```python
def build_context(retrieved):
    """将检索结果组装为 Prompt 上下文"""
    parts, citations = [], []
    for i, item in enumerate(retrieved, 1):
        stock   = item.get("stock_code", "")
        year    = item.get("year", "")
        page    = item.get("page_num", "")
        section = item.get("section", "")

        # 构建来源标签：如 "[1] 600519 2023年报 · 第三章 · 第38页"
        label = f"[{i}] {stock} {year}年报"
        if section: label += f" · {section}"
        if page and page != -1: label += f" · 第{page}页"

        # 层级分块优先用父块内容（信息更完整）
        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label,
                          "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations
```

#### System Prompt 设计

```python
SYSTEM_PROMPT = """你是一个专业的财务分析助手，专门回答关于中国上市公司年度报告的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营业收入为1476亿元[1]
4. 数字要精确，不得四舍五入或模糊表达
5. 回答简洁，重点突出，避免无关废话"""
```

**设计考量**：

- **规则 1**（防幻觉）：强制约束只能使用检索到的资料
- **规则 2**（拒答机制）：防止模型在信息不足时编造答案
- **规则 3**（可溯源性）：`[1]` 标记让用户能追溯到出处
- **规则 4**（精确性）：财务分析场景对数字精度要求高
- **规则 5**（简洁性）：回答聚焦核心问题

#### LLM 调用

```python
def call_llm(query, context, client):
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model="qwen3.7-plus",         # 可换 qwen-turbo（更快）/ qwen-max（更强）
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,              # 低温度保证事实准确性
    )
    return resp.choices[0].message.content
```

**为什么 `temperature=0.1`**：财务数据问答需要准确性而非创造性。低温度让模型更倾向于选择概率最高的 token，减少随机性，降低编造数字的风险。

---

## 四、两条流程的关系全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                     离线构建（一次性/定期）                         │
│                                                                   │
│  PDF  TXT  DOCX  MD  HTML  CSV  XLSX  PNG                        │
│   │    │     │    │    │    │     │     │                          │
│   └────┴─────┴────┴────┴────┴─────┴─────┘                          │
│                      │                                            │
│                      ▼                                            │
│  format_loaders/  (8个loader → 统一 ParsedBlock)                   │
│                      │                                            │
│                      ▼                                            │
│  data/parsed/*.json  (结构化 JSON, 统一格式)                       │
│                      │                                            │
│                      ▼                                            │
│  chunk_documents.py  (3种策略: fixed/semantic/hierarchical)        │
│                      │                                            │
│                      ▼                                            │
│  data/chunks/all_semantic.json                                    │
│                      │                                            │
│                      ▼                                            │
│  build_index.py  (DashScope embedding → FAISS IndexFlatIP)        │
│                      │                                            │
│                      ▼                                            │
│  vectorstore/faiss_semantic/                                      │
│    ├── faiss_index.bin  (~41MB, 10353×1024 维向量)                │
│    └── faiss_meta.json  (~15MB, 文本+元信息)                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 加载到内存
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     在线检索（每次查询）                            │
│                                                                   │
│  用户问题: "茅台2023年营收多少"                                     │
│         │                                                         │
│         ▼                                                         │
│  ① query_rewrite (qwen-turbo, 可选)                                │
│     "茅台最近怎么样" → "贵州茅台2023年营业收入净利润同比增长率"       │
│         │                                                         │
│         ▼                                                         │
│  ② VectorStore.search()  ────  FAISS 内积搜索 top-10              │
│     DashScope embedding (1次API) + 暴力内积扫描 (<1ms)             │
│         │                                                         │
│         ├─── ③ BM25Store.search()  ──  jieba分词 + BM25 top-10    │
│         │     对精确数字/代码/术语的召回优于纯向量检索               │
│         │         │                                               │
│         └────┬────┘                                               │
│              ▼                                                    │
│  ④ reciprocal_rank_fusion()  ──  RRF 融合排名                    │
│     score(d) = Σ 1/(60 + rank_i(d)), 排名融合无需训练             │
│              │                                                    │
│              ▼                                                    │
│  ⑤ rerank (CrossEncoder, 可选)  ──  top-10 → top-4              │
│     bge-reranker-base 双向注意力精排, 精度显著优于纯向量排序        │
│              │                                                    │
│              ▼                                                    │
│  ⑥ 阈值检查 (vec_score < 0.25 → 拒绝回答)                          │
│     始终用余弦相似度分数, RRF分数量纲不适用                         │
│              │                                                    │
│              ▼                                                    │
│  ⑦ build_context() → call_llm (qwen3.7-plus, temperature=0.1)    │
│     标签 + parent_content, 来源标注, 防幻觉约束                    │
│              │                                                    │
│              ▼                                                    │
│  答案: "贵州茅台2023年营业收入为1476亿元[1]，同比增长19%[1]"         │
│  引用: [1] 600519 2023年报 · 第三章 · 第38页                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、关键设计决策

### 离线构建

| 决策 | 说明 |
|------|------|
| **统一 ParsedBlock 格式** | 下游 chunk/index/query 完全不需要感知原始文件格式 |
| **懒加载 loader** | TXT 解析不需要 pdfplumber，PDF 解析不需要 python-docx，各格式依赖独立 |
| **表格独立成块** | 表格与文字混合会降低 embedding 质量——LLM 对 Markdown 表格的理解远超原始数组 |
| **表格转为 Markdown** | LLM 预训练数据大量包含 Markdown 表格，理解能力远强于其他格式 |
| **`page_num=0` 表示无分页** | 与 PDF 的真实页码区分，UI 层自动隐藏页码信息 |
| **L2 归一化 + IndexFlatIP** | 内积等价于余弦相似度，分数 0~1 物理意义明确，适合设阈值 |
| **向量与元数据分离存储** | FAISS 高效二进制 + JSON 可读可编辑，互不干扰 |
| **BATCH_SIZE=10** | DashScope text-embedding-v3 单批次实际上限为 10（非文档所述的 25），踩坑验证 |

### 在线检索

| 决策 | 说明 |
|------|------|
| **BM25 + 向量双路检索** | 关键词匹配补语义检索的盲区，对精确数字/代码/专有术语尤其重要 |
| **RRF 而非加权融合** | 不同检索器分数量纲不同（0~1 vs 0~50+），排名融合更稳健，无需训练权重 |
| **CrossEncoder 精排** | Bi-Encoder 粗筛 10 个（全库扫描快）+ CrossEncoder 精排 4 个（双向注意力准） |
| **`vec_score` 做阈值** | RRF 分数量纲不可比（~0.016），余弦相似度 0~1 才有物理意义 |
| **指定过滤器时跳过阈值** | 用户主动缩小范围说明对数据有预期，即使相似度低也应尝试回答 |
| **`temperature=0.1`** | 财务数据问答需要准确性，低温度减少编造数字的风险 |
| **`parent_content` 优先** | 层级分块时检索用小块（精准），生成时给 LLM 读大块（信息完整） |
| **`qwen-turbo` 改写 + `qwen3.7-plus` 生成** | 改写只需快和便宜，生成需要质量和推理能力——不同任务不同模型 |

### 防幻觉体系

这是一个**多层防御**的设计：

1. **检索层**：双路检索 + RRF 融合 → 确保召回率和精度
2. **重排序层**：CrossEncoder 精排 → 确保最相关的排在前面
3. **阈值层**：`vec_score < 0.25` → 拒绝回答 → 防止基于弱相关内容的胡说
4. **Prompt 层**：System Prompt 约束 → "只根据参考资料回答" + "信息不足则拒绝"
5. **参数层**：`temperature=0.1` → 降低随机性 → 减少编造

---

## 六、使用方式

### 离线构建

```bash
# 1. 解析多格式文档
python src/parse_documents.py                    # 扫描默认目录
python src/parse_documents.py --dir /path/to/docs  # 指定目录
python src/parse_documents.py --format txt,csv   # 过滤格式

# 2. 分块（修改 chunk_documents.py 中的 STRATEGY 变量切换策略）
python src/chunk_documents.py

# 3. 构建索引
export DASHSCOPE_API_KEY="sk-xxx"
python src/build_index.py
```

### 在线检索

```bash
# 交互式
python src/rag_pipeline.py

# 单次查询
python src/rag_pipeline.py --query "茅台2023年营收"

# 带过滤
python src/rag_pipeline.py --query "营收情况" --stock 600519 --year 2023

# 开启查询改写
python src/rag_pipeline.py --query "茅台怎么样" --query-rewrite

# 消融实验（关闭 BM25 和 Rerank）
python src/rag_pipeline.py --query "..." --no-bm25 --no-rerank
```

### HTTP 服务

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000 使用可视化页面
# POST /query         标准问答
# POST /query/debug   调试接口（逐步骤返回中间结果）
```
