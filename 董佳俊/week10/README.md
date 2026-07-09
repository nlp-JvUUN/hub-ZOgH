# 英雄联盟 RAG 问答系统

> Week 10 作业 — 基于自建知识库的检索增强生成问答系统

---

## 项目定位

以"英雄联盟知识问答"为场景，构建一套轻量级 RAG（检索增强生成）系统。与课程项目的上市公司年报问答形成对比：同样的流水线设计思路，不同的数据源、向量化方式和 LLM 后端。

核心特点：
- **纯本地向量化**：字符级 bigram + TF-IDF，不依赖任何外部 embedding API
- **纯标准库 + numpy**：除 numpy 外无需 pip 安装任何包
- **DeepSeek 作为 LLM**：通过 OpenAI 兼容接口调用 `deepseek-chat`

---

## 技术架构

```
自建知识文档 (10篇 .txt)
       │
       ▼  generate_data.py
   数据生成
       │
       ▼  chunk_docs.py
   文档分块 (500字/块, overlap=50)
       │
       ▼  build_index.py
   TF-IDF 向量化 → 向量存储
       │
       ▼  rag_qa.py
   查询向量化 → cosine 检索 → DeepSeek LLM 生成
```

---

## 目录结构

```
homework/
├── README.md              # 本文档
├── USAGE.md               # 使用指南
├── DESIGN.md              # 设计决策与踩坑记录
├── data/
│   ├── raw/               # 10 篇英雄联盟知识文档
│   ├── chunks/            # 分块后的 JSON
│   └── parsed/            # 预留
├── vectorstore/           # TF-IDF 向量 + 词汇表 + 元数据
└── src/
    ├── generate_data.py   # 生成知识库文档
    ├── chunk_docs.py      # 文档分块
    ├── build_index.py     # TF-IDF 向量索引构建
    ├── download_data.py   # 维基百科采集（备用，需网络）
    └── rag_qa.py          # 问答流水线
```

---

## 与课程项目的关键差异

| 维度 | 课程项目 | 本作业 |
|------|---------|--------|
| 数据源 | 巨潮资讯网 15 份年报 PDF | 自建 10 篇中文知识文档 |
| 文档解析 | pdfplumber + PyMuPDF + OCR | 纯文本，无需解析 |
| Embedding | DashScope text-embedding-v3 (1024维) | 字符级 bigram + TF-IDF (6804维) |
| 向量存储 | FAISS IndexFlatIP | numpy 内存数组 + 本地文件 |
| 检索策略 | 向量 + BM25 → RRF 融合 → Rerank | 单路 TF-IDF cosine 检索 |
| LLM | DashScope qwen-plus | DeepSeek deepseek-chat |
| 评估 | RAGAS 四项指标 + 消融实验 | 未包含 |
| HTTP 服务 | FastAPI / Swagger | 未包含 |
| 依赖复杂度 | 需安装 10+ 个包 | numpy + 标准库 |
