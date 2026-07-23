# 使用指南

> 英雄联盟 RAG 问答系统 — 完整操作流程

---

## 环境要求

- Python 3.9+（需 numpy）
- DeepSeek API Key（环境变量 `DEEPSEEK_API_KEY`）

当前工作区使用的 Python 路径：
```
C:\Users\86188\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

---

## 完整流程（首次使用）

### 步骤 1：生成知识库数据

```powershell
$py = "C:\Users\86188\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py homework\src\generate_data.py
```

产出 10 篇 txt 文档到 `homework/data/raw/`，覆盖英雄背景、赛事、地图机制等主题。

### 步骤 2：文档分块

```powershell
& $py homework\src\chunk_docs.py
```

参数（修改脚本顶部变量）：
- `CHUNK_SIZE = 500`（每块字符数）
- `OVERLAP = 50`（重叠字符数）

产出 `homework/data/chunks/all_chunks.json`。

### 步骤 3：构建向量索引

```powershell
& $py homework\src\build_index.py
```

内部流程：
1. 加载所有 chunk
2. 对每个 chunk 做字符级 unigram + bigram 分词
3. 构建词汇表，过滤出现次数过少/过多的 term
4. 计算 TF-IDF 矩阵，L2 归一化
5. 保存向量和词汇表

### 步骤 4：问答

设置环境变量：
```powershell
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"
```

单次查询：
```powershell
& $py homework\src\rag_qa.py --query "亚索的大招是什么"
```

显示检索过程：
```powershell
& $py homework\src\rag_qa.py --query "李青的回旋踢怎么操作" --verbose
```

交互式模式：
```powershell
& $py homework\src\rag_qa.py
```

---

## 问答流程详解

```
用户输入 "LPL赛区拿过几次冠军"
       │
       ▼  问句向量化（同构建索引时的分词方式）
   unigram + bigram 分词 → TF 向量 → L2 归一化
       │
       ▼  cosine 相似度检索
   查询向量 · 文档矩阵 → top-4 最相似 chunk
       │
       ▼  阈值检查（score > 0.15 继续，否则拒绝回答）
       │
       ▼  DeepSeek deepseek-chat 生成
   拼接上下文 + 系统提示 + 问题 → LLM → 带来源标注的回答
```

---

## 代码中直接调用

```python
import sys
sys.path.insert(0, "homework/src")
from rag_qa import RAGPipeline

pipeline = RAGPipeline()  # 加载索引（首次约需 1 秒）
result = pipeline.query("亚索的被动技能是什么")

print(result["answer"])       # LLM 生成的回答
print(result["sources"])      # 来源列表 [{"index": 1, "source": "...", "score": 0.19}]
```

---

## 扩展方向

- **换 LLM**：修改 `rag_qa.py` 中的 `DEEPSEEK_URL` 和 `LLM_MODEL` 即可切换到任何 OpenAI 兼容接口
- **增加数据**：往 `generate_data.py` 的 `DOCUMENTS` 列表里加条目，重新跑步骤 1→2→3
- **改用语义向量**：将 `build_index.py` 中的 TF-IDF 替换为 DashScope API 调用
- **添加 BM25**：安装 `jieba` + `rank_bm25`，在 `rag_qa.py` 的 `search()` 方法中加入关键词检索路径
