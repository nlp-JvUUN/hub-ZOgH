# 船舶术语 RAG 问答系统 — 课后作业

基于 `rag_annual_report` 项目架构，将数据域从**上市公司年报**切换为**船舶/验船术语知识库**，构建一套完整的 RAG 问答系统。

## 作业目标

1. 掌握 RAG 完整流水线：数据解析 → 文档分块 → 向量索引 → 检索生成 → 评估
2. 理解三种分块策略（fixed / semantic / hierarchical）的差异
3. 理解混合检索（向量 + BM25 + RRF）的优势
4. 学会设计评测题集和消融实验

## 项目结构

```
ship_rag_homework/
├── data/
│   ├── raw_csv/              # 原始训练资料（术语类120个随机问题评分.csv + 问答对.csv）
│   ├── parsed/               # parse_csv.py 输出（结构化 JSON）
│   ├── chunks/               # chunk_documents.py 输出（三种策略）
│   └── manifest.json         # 数据索引
├── vectorstore/              # build_index.py 输出（FAISS 索引 + 元数据）
├── src/
│   ├── parse_csv.py          # 模块一：CSV → 结构化 JSON
│   ├── chunk_documents.py    # 模块二：三种分块策略
│   ├── build_index.py        # 模块三：向量索引构建
│   └── rag_pipeline.py       # 模块四：RAG 问答流水线
├── evaluation/
│   ├── questions.json        # 20 道标准测试题
│   ├── evaluate.py           # 评估脚本（Hit Rate + MRR + 答案匹配率）
│   └── compare_strategies.py # 消融实验脚本
├── requirements.txt
└── README.md                 # 本文件
```

## 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key（必须）
export DASHSCOPE_API_KEY="sk-xxx"  # Linux/Mac
# 或
set DASHSCOPE_API_KEY=sk-xxx       # Windows
```

## 运行步骤

### 第一步：数据解析（模块一）

```bash
python src/parse_csv.py
```

将 CSV 训练资料转换为结构化 JSON，输出到 `data/parsed/`。

### 第二步：文档分块（模块二）

```bash
# 默认使用语义分块（semantic）
python src/chunk_documents.py
```

修改 `src/chunk_documents.py` 中的 `STRATEGY` 变量：
- `"fixed"` — 固定大小分块
- `"semantic"` — 语义分块（默认）
- `"hierarchical"` — 层级分块

### 第三步：构建向量索引（模块三）

```bash
python src/build_index.py
```

使用 DashScope `text-embedding-v3` 计算 embedding，构建 FAISS 索引。

### 第四步：问答测试（模块四）

```bash
# 交互式问答
python src/rag_pipeline.py

# 单条测试
python src/rag_pipeline.py --query "Cathodic Protection 是什么"

# 按类别过滤
python src/rag_pipeline.py --query "..." --category 术语类

# 开启查询改写
python src/rag_pipeline.py --query "..." --query-rewrite

# 消融测试（关闭 BM25）
python src/rag_pipeline.py --query "..." --no-bm25
```

### 第五步：评估（模块五）

```bash
# 运行全部 20 道测试题
cd evaluation
python evaluate.py

# 只跑部分题
python evaluate.py --question-ids 1,2,3

# 消融实验
python compare_strategies.py
python compare_strategies.py --strategies semantic,hierarchical --modes vector_only,hybrid
```

## 评分标准

| 模块 | 内容 | 分值 |
|------|------|------|
| 模块一 | parse_csv.py 正确解析两种 CSV 格式 | 20分 |
| 模块二 | 实现三种分块策略，语义分块保留问答对完整性 | 20分 |
| 模块三 | 成功构建 FAISS 索引，支持元数据过滤 | 15分 |
| 模块四 | RAG 流水线完整（向量+BM25+RRF+LLM），支持查询改写和过滤 | 25分 |
| 模块五 | 20道测试题设计合理，评估指标计算正确 | 20分 |
| **加分项** | 实现 Web 界面 / 支持多轮对话 / 额外优化 | +10分 |

## 关键技术对比

| 环节 | 年报 RAG | 船舶术语 RAG |
|------|---------|-------------|
| 数据源 | PDF 年报 | CSV 问答对 |
| 解析重点 | 表格+章节结构 | 问答对结构化 |
| 分块策略 | 语义/固定/层级 | 以问答对边界为核心 |
| 检索特点 | 数字精确匹配 | 术语中英混用匹配 |
| Prompt | 财务分析助手 | 船舶验船术语助手 |

## 参考文档

- 原项目架构：`../week10 检索增强生成RAG/rag_annual_report/ARCHITECTURE.md`
- 训练资料：`data/raw_csv/术语类120个随机问题评分.csv`、`data/raw_csv/问答对.csv`

## 常见问题

**Q: 没有 DashScope API Key 怎么办？**
A: 可以改用本地 BGE 模型（参考原项目 `src_langchain/build_index_lc.py`）。

**Q: 索引构建失败？**
A: 确保已运行 `parse_csv.py` 和 `chunk_documents.py`，且 `data/chunks/all_semantic.json` 存在。

**Q: 如何切换分块策略做对比？**
A: 修改 `chunk_documents.py` 中的 `STRATEGY` 变量，重新运行 `chunk_documents.py` 和 `build_index.py`。

---

**作业提交**：代码 + README + 消融实验结果截图
