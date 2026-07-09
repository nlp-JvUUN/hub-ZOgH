# 设计决策与踩坑记录

> 英雄联盟 RAG 问答系统 — 方案选型、决策原因与经验总结

---

## 一、为什么用 TF-IDF 而不是语义 Embedding

### 选型背景

本作业使用的 Python 环境是一个精简的 bundled runtime，仅包含 `numpy` + 标准库，无法安装 `sentence-transformers`、`openai` 等第三方包。DashScope embedding API 虽然可用但需要额外网络开销。

### TF-IDF 方案的可行性

对于中文文本，字符级 bigram 可以捕获相当多的语义信息：

```
"亚索的大招狂风绝息斩" → 分詞为 [亚,索,的,大,招,狂,风,绝,息,斩, 亚索,索的,的大,大招,招狂,狂风,风绝,绝息,息斩]
```

这些 bigram 中，"狂风"、"绝息"、"大招"、"亚索" 都是高信息量的特征。配合 IDF 降权常见词、TF 归一化，在 29 个 chunk 的小规模语料上，检索精度足够用。

### 实测效果

```python
# 查询 "亚索的大招是什么"
# top-1 得分: 0.190
# top-1 内容: "...R 技能——狂风绝息斩..."
# → 命中正确 chunk
```

cosine 得分偏低（0.15~0.45 区间）是因为 6804 维稀疏向量在高维空间中 cosine 值天然偏低——这属于正常现象，阈值调为 0.15 即可。

### 局限

- 无法理解同义词（"回旋踢" 和 "Insec Kick" 在 bigram 层面完全不同，依赖语料中同时出现两者）
- 维度随语料增长，大规模场景不适用
- 没有语义泛化能力

---

## 二、为什么用 DeepSeek 而不是 DashScope

| 维度 | DashScope qwen-plus | DeepSeek deepseek-chat |
|------|-------------------|----------------------|
| 可用性 | 需要单独注册阿里云账号 | 用户已有 API Key |
| 性能 | 中文理解强，温度稳定 | 同级别中文能力 |
| 接口兼容 | OpenAI 兼容 | OpenAI 兼容 |
| 费用 | 按量计费 | 按量计费 |

本次作业选择 DeepSeek 纯粹是因为用户环境已有配置，减少额外的账号和密钥管理工作。两个模型在这个规模的问答任务上表现没有实质差异。

---

## 三、为什么用自建数据而不是网络抓取

### 原始计划

最初设计了一个 `download_data.py` 脚本，通过维基百科 API 抓取英雄联盟相关页面的文本内容。包含 16 个中文维基页面，覆盖游戏概述、角色、赛事、机制等。

### 遇到的问题

沙箱环境中无法建立到 `zh.wikipedia.org` 的 HTTPS 连接（`TimeoutError: timed out`）。尝试了以下方法均无效：
- `urllib.request` 直连
- `requests` 库
- 带 User-Agent 头的请求
- 使用维基百科 API 的 `explaintext` 和 `format=json` 参数

根本原因：沙箱网络策略限制了对外部 HTTPS 的访问。

### 替代方案

改为在 `generate_data.py` 中直接生成高质量的中文知识文档。优点：
- 不受网络限制，100% 可运行
- 内容可控，可以针对常见问答设计文档结构
- 文档间有自然的关联关系（不同英雄的故事有交集），适合测试多文档综合问答

缺点：
- 数据量有限（10 篇，约 11,000 字）
- 内容深度依赖预设

---

## 四、Pipeline 设计取舍

### 保留的课程设计

- **分阶段流水线**：`generate → chunk → index → query` 四步分离，每步独立可运行
- **数据结构**：chunk 的 `content` + `metadata` 结构，与课程项目保持一致
- **命令行接口**：支持 `--query` 单次提问和交互式模式

### 简化的部分

- **无 PDF 解析**：自建文本无需提取层，省掉了 pdfplumber + PyMuPDF + OCR 的复杂管线
- **无 BM25 检索**：`rank_bm25` 和 `jieba` 在 runtime 中不可用，且 29 个 chunk 的规模下单路检索足够
- **无 Rerank**：候选集太小（仅 29），二次精排的边际收益为零
- **无评估体系**：作业要求是"能用代码做出回答"，评估作为加分项未包含
- **无 HTTP 服务**：同上，命令行交互满足作业要求

### 新增的设计

- **完全自包含**：所有依赖仅限于 numpy + 标准库，不需要任何 pip install
- **词汇表持久化**：`vocab.json` 保存了词汇表，查询时无需重新构建
- **降级策略**：当 `SCORE_THRESHOLD` 未被任何 chunk 满足时，系统明确表示"无法回答"

---

## 五、踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| pip install 超时 | 沙箱限制 PyPI 网络 | 放弃安装第三方包，改用自带 numpy |
| urllib 请求 Wikipedia 超时 | 沙箱限制外网 HTTPS | 改为自建数据源 |
| miniconda Python 无 numpy | conda base 环境不含科学计算包 | 改用 bundled Python runtime |
| bundled Python 无 requests | 精简 runtime 未打包 requests | 全部改用 `urllib.request`（标准库） |
| apply_patch 无法处理多行字符串中的空行 | patch 格式将空行解析为 hunk 分隔符 | 改用 PowerShell here-string + Out-File 写入 |
| TF-IDF cosine 得分偏低 | 6804 维稀疏向量的高维特性 | 将 `SCORE_THRESHOLD` 从 0.35 降为 0.15 |

---

## 六、文档结构参考

本作业参考课程项目的文档体系，建立了三份说明文档：

| 文档 | 课程对应 | 内容 |
|------|---------|------|
| `README.md` | 项目总览 | 定位、架构、与课程对比 |
| `USAGE.md` | `USAGE_GUIDE.md` | 环境准备、操作流程、API 调用示例 |
| `DESIGN.md` | `ARCHITECTURE.md` + `PROJECT_LOG.md` | 选型决策、设计取舍、踩坑记录 |

未包含 `RESUME_GUIDE.md`（简历指南），因为作业场景不需要。
