# week10 — 基于手头文件的问答系统（最小逻辑闭环）

> 作业要求：基于手头文件写一个问答系统，不需要网页，能用代码做出回答即可，方法任选。
> 本作业选择**老师课件**（`RAG.pptx` 47 页 + `ragas.pdf` 8 页）作为"手头文件"，
> 实现一个 **纯 Python 标准库、零第三方依赖** 的检索式问答系统，
> 并保留"设置 API Key 后自动升级为 LLM 生成答案"的接口。

---

## 一、主题与设计思路

### 1. 为什么选课件当"手头文件"

本周主题就是 RAG，用 RAG 课件本身做知识库，形成闭环：**问答系统回答"什么是 RAG"**。
所有问题的答案都能溯源到课件具体页码，回答效果一眼可验证。

### 2. 最小逻辑闭环（对应课件 Part 2/3/4）

```
【离线】直接解析手头文件原件 → 分块 → 知识库
   build_knowledge.py
   RAG.pptx(47页) ──┐
                     ├─→ 按"页"切分 + 按 700 字符合并 ──→ data/knowledge.json（102 块）
   ragas.pdf(8页) ──┘
   （PPTX 用 zipfile+XML 直接解析，零依赖；PDF 需可选安装 pymupdf）

【在线】问题 → 检索 → 回答
   qa_system.py
   问题 ──→ 分词 ──→ BM25 打分 ──→ top-k ──→ 相关性门槛 ──→ 回答
                                                  ├─ 保底：返回原文片段（带来源页码）
                                                  └─ 可选：调 LLM 生成带引用的答案
```

| 环节 | 本作业实现 | 对应课件知识点 |
|------|-----------|---------------|
| 读取 | **PPTX 直接解析**（zip 内 slide XML 的 `<a:t>` 文本，等价 python-pptx 底层）；PDF 可选（pymupdf）；txt/md 直接读 | Part 2 数据源/Loader 选型 |
| 分块 | 页边界 + 700 字符段落合并（过滤页脚噪声） | Part 2 分块策略（块太大/太小都不好）|
| 检索 | 自实现 BM25（TF + IDF + 长度归一化）| Part 3 BM25 关键词检索 |
| 防幻觉 | 相关性门槛：无有效词命中 → 拒绝回答 | Part 4 Prompt 设计/拒绝条款 |
| 回答 | 原文片段（保底）/ LLM 生成（可选） | Part 4 增强生成 + 来源引用 |
| 评估 | `--selftest`：9 题 Hit@3 + 1 题越界拒绝 | Part 5 检索质量指标 |

### 3. 为什么"零依赖"也能回答

- **分词**：中文按"单字 + 相邻二字组"，英文按整词（`qa_system.py` 里约 10 行）
- **BM25**：标准 Okapi BM25 公式自实现（约 40 行，`k1=1.5, b=0.75`）
- **LLM 可选**：装了 `openai` 且有 Key 时自动启用；调用失败自动降级回原文模式

---

## 二、目录结构

```
week10/
├── README.md               # 本文档
├── build_knowledge.py      # 第 1 步：直接解析手头文件原件 → 分块 → knowledge.json
├── qa_system.py            # 第 2 步：检索 + 回答（含自测）
└── data/
    ├── RAG.pptx            # 手头文件原件①：老师课件（47 页，直接解析）
    ├── ragas.pdf           # 手头文件原件②：RAGAS 论文（8 页，直接解析，需 pymupdf）
    ├── raw_课程资料.txt     # 备选：已提取的文本版（无 pymupdf 时兜底）
    └── knowledge.json      # 知识库（102 块，由 build_knowledge.py 生成）
```

## 三、使用方法

```bash
# 1. 构建知识库（只需做一次；默认直接解析课件原件）
python build_knowledge.py
#    → 解析 data/RAG.pptx + data/ragas.pdf（读 PDF 需: pip install pymupdf）
#    只装了解析 PPT：python build_knowledge.py --input data/RAG.pptx
#    换自己的文件：  python build_knowledge.py --input 任意.pptx / 年报.pdf / 笔记.txt

# 2a. 交互式问答
python qa_system.py

# 2b. 单次提问
python qa_system.py --query "什么是RAG？"

# 2c. 本地自测（不调 API，不花钱）
python qa_system.py --selftest

# 2d. 启用 LLM 生成（可选，任选一个 Key）
export DASHSCOPE_API_KEY=sk-xxx
# 或: export OPENAI_API_KEY=sk-xxx
# 或换 DeepSeek: export LLM_BASE_URL=https://api.deepseek.com LLM_MODEL=deepseek-chat
python qa_system.py --query "什么是RAG？"
```

## 四、运行示例（真实输出）

**示例 1：BM25 打分主要考虑哪三个因素？**（原文片段模式，可溯源）

```
问题：BM25打分主要考虑哪三个因素？
[1] [RAG.pptx] 第18页
Part 3 · 检索技术
BM25：老而弥坚的关键词检索
在向量检索大热之前，BM25 统治搜索引擎20年——它依然有独特价值
BM25 直觉理解
BM25 打分的核心逻辑：
📈 词频（TF）    同一词出现越多次，越相关，但有上限——防止垃圾词刷高分
📉 逆文档频率（IDF）「的」「是」出现在每篇文档 → 权重极低；稀有词权重高
📏 文档长度归一化  长文档自然出现更多词 → 按文档长度调整得分
...
── 来源 ──
  [RAG.pptx] 第18页  (score=7.09)
```

**示例 2：越界问题（防幻觉）**

```
问题：贵州茅台2023年营业收入是多少？
根据知识库未能找到与该问题相关的内容。
```

知识库里没有茅台财报 → 系统明确拒绝，而不是乱编（对应课件 Part 1 的幻觉问题）。

**示例 3：查看检索过程**

```
python qa_system.py --query "什么是Graph RAG" --verbose
[INFO] 检索: 命中 3 块 | 最高分 7.8099 | 分词: 什/么/是/什么/么是/graph/rag
```

## 五、自测结果（`python qa_system.py --selftest`）

10 道题：9 道内容题（Hit@3 = 期望来源页出现在检索结果中）+ 1 道越界题（期望拒绝）。

```
[HIT ] 基础概念 | 什么是RAG？RAG的三个核心步骤是什么？        → 第6页   ✓
[HIT ] 动机理解 | 大模型为什么会产生幻觉？                    → 第4页   ✓
[HIT ] 分块细节 | 文本分块时chunk overlap有什么作用？          → 第11页  ✓
[HIT ] 检索原理 | BM25打分主要考虑哪三个因素？                → 第18页  ✓
[HIT ] 混合检索 | RRF混合检索的公式是什么？                   → 第19页  ✓
[HIT ] 评估指标 | RAGAS中Faithfulness指标衡量什么？           → 第30页  ✓
[HIT ] 评估框架 | RAGAS框架是什么？怎么用？                   → 第33页  ✓
[HIT ] 选型知识 | 课件推荐的中文Embedding模型有哪些？         → 第13页  ✓
[HIT ] 进阶架构 | 什么是Graph RAG？适合什么场景？             → 第40页  ✓
[PASS] 越界问题 | 贵州茅台2023年营业收入是多少？              → 拒绝    ✓

结果: Hit@3 = 9/9  |  越界拒绝 = 1/1
```

## 六、关键设计说明（写报告/答辩用）

1. **直接解析课件原件**：`build_knowledge.py` 不依赖任何解析库也能吃 PPTX——
   PPTX 本质是 zip 包，用标准库 `zipfile` + `ElementTree` 读取 `ppt/slides/slideN.xml`
   里的 `<a:t>` 文本节点（这正是 python-pptx 的底层机制）；PDF 通过可选安装的
   pymupdf 逐页提取。回答里的"第 N 页"就是课件的真实页码。
2. **相关性门槛（防幻觉第一道闸门）**：检索必须命中至少一个"非纯数字的有效词"
   （二字组/英文词）。单字噪声大（"入"既匹配"收入"也匹配"输入"），纯数字
   （如"2023"）在参考文献里到处都是，都不能单独构成相关性。
3. **有效词只取长度 ≥ 2 的 token**：既消除单字噪声，又让打分集中在有判别力的词上。
4. **LLM 模式自动降级**：没 Key 用原文片段；有 Key 用大模型生成；调用失败自动降级，
   保证"任何时候都能跑"。
5. **可溯源**：每块保留 `[来源] 第N页` 标记，答案（无论哪种模式）都带出处。

## 七、可升级方向（对应课件 Part 3/5/6）

| 升级点 | 对应课件 |
|--------|---------|
| 加向量检索（本地 BGE embedding + FAISS），与 BM25 做 RRF 混合 | Part 2/3 混合检索 |
| 加 Rerank（bge-reranker），k=10 → 精排 top-3 | Part 3 重排序 |
| 用 RAGAS 跑 Faithfulness / Answer Relevancy 等指标 | Part 5 评估体系 |
| 查询改写（Multi-Query / 重写），处理模糊问题 | Part 6 Pre-Retrieval |
| 换更大的手头文件（PDF 直接入库，需 pymupdf） | Part 2 Loader 选型 |
