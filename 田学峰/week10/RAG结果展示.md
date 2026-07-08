# RAG结果展示

# 1 数据

采用的是文本分类任务、序列标注任务和文本匹配总结的相关知识点所得的pdf文档
根据上述知识文档做解析和chunking

# 2 执行

```bash
e:\ai\week10检索增强生成练习\src>python -m uvicorn serve:app --host 0.0.0.0 --port 8000
INFO:     Started server process [23176]
INFO:     Waiting for application startup.
2026-07-08 23:10:36,866 [INFO] 服务启动，初始化 RAG Pipeline...
2026-07-08 23:10:37,760 [INFO] Loading faiss with AVX512 support.
2026-07-08 23:10:37,761 [INFO] Could not load library with AVX512 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx512'")
2026-07-08 23:10:37,761 [INFO] Loading faiss with AVX2 support.
2026-07-08 23:10:37,761 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 23:10:37,761 [INFO] Loading faiss.
2026-07-08 23:10:38,024 [INFO] Successfully loaded faiss.
2026-07-08 23:10:40,339 [INFO] FAISS 索引加载完成，共 472 条向量
C:\Users\12267\AppData\Roaming\Python\Python313\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 23:10:40,641 [INFO] 构建 BM25 索引（分词中，请稍候）...
Building prefix dict from the default dictionary ...
2026-07-08 23:10:40,642 [DEBUG] Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\12267\AppData\Local\Temp\jieba.cache
2026-07-08 23:10:40,643 [DEBUG] Loading model from cache C:\Users\12267\AppData\Local\Temp\jieba.cache
Loading model cost 0.633 seconds.
2026-07-08 23:10:41,275 [DEBUG] Loading model cost 0.633 seconds.
Prefix dict has been built successfully.
2026-07-08 23:10:41,275 [DEBUG] Prefix dict has been built successfully.
2026-07-08 23:10:41,550 [INFO] BM25 索引完成
2026-07-08 23:10:41,551 [INFO] Pipeline 初始化完成，开始接受请求
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:59725 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:59725 - "GET /favicon.ico HTTP/1.1" 404 Not Found
2026-07-08 23:12:13,426 [INFO] NumExpr defaulting to 8 threads.
2026-07-08 23:12:18,635 [INFO] 加载本地 Embedding 模型（sentence-transformers）: e:\ai\week10检索增强生成练习\models\bge-small-zh-v1.5
2026-07-08 23:12:18,638 [INFO] No device provided, using cpu
2026-07-08 23:12:18,647 [INFO] Loading SentenceTransformer model from e:\ai\week10检索增强生成练习\models\bge-small-zh-v1.5.
Loading weights: 100%|████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 254.02it/s]
Batches: 100%|███████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.27s/it]
2026-07-08 23:12:38,066 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
INFO:     127.0.0.1:60147 - "POST /query/debug HTTP/1.1" 200 OK
```

# 3 结果

调用LLM模型qwen-plus得到一下结果：

```http
命名实体识别（NER）是一种**序列标注任务**，其核心目标是对文本中每个词元（token）分配一个标签，以识别出具有特定语义类型的命名实体（如人名、地名、机构名等），并标定其实体边界与类别[1]。例如，在句子中，“马云”被识别为人名（PER），其中“马”标记为起始（B-PER），“云”标记为内部（I-PER）；“阿里巴巴”被识别为机构名（ORG），对应标签为 B-ORG 和 I-ORG；而“创立”“了”等非实体词则标记为 O（Outside）[1]。

从更本质的角度看，NER 并非仅限于传统的人名、地名、机构名识别，而是一个**通用的 span 分类器**：只要定义好标签体系并提供足够标注样本，模型就能学习在给定上下文中判断某一段连续 token 是否属于目标类别——该能力与具体实体类型名称无关[3]。

关于主流方法，参考资料中**未明确列举或介绍具体的模型方法（如 CRF、BiLSTM-CRF、BERT-NER 等）**，也未涉及算法原理、架构演进或训练策略等内容。因此，根据所提供的资料，**无法回答“有哪些主流方法”这一部分**。

✅ 可确认的信息总结如下：
- NER 是序列标注任务，逐 token 打标签（如 B-PER、I-ORG、O）[1]；
- 其本质是通用 span 分类器，强调上下文驱动的片段分类能力，而非依赖预设实体类型的先验知识[3]；
- NER 的标签具有稳定性（如“马云”在任何文本中都应标为 PER）[4]；
- 区别于语义角色标注（SRL），NER 不关心实体在句中的谓词论元角色（如 ARG0/ARG1），只关注“是什么实体”[2]。

❌ 主流方法（如基于统计的 HMM、CRF，或基于深度学习的 BiLSTM-CRF、Transformer-based NER 模型等）在参考资料中**未被提及**，故根据现有材料无法作答。

> **结论**：  
> 命名实体识别（NER）是一种序列标注任务，旨在识别文本中具有特定语义类型的命名实体，并为其标注边界与类别[1]；其本质是通用 span 分类器，具备跨类型泛化潜力[3]。  
> 关于主流方法，**根据提供的资料无法回答此问题**。
来源引用
[1] 《NLP序列标注任务详解》 · 序列标注 · 第一章 序列标注基础 > 1.1 核心定义 > 什么是序列标注？ > 示例 1：命名实体识别（NER） · 第2页
[2] 《NLP序列标注任务详解》 · 序列标注 · 第二章 NLP 中的典型应用 > 2.1 中文分词（Tokenization） > 标签 > SRL 的业务价值 · 第10页
[3] 《NLP序列标注任务详解》 · 序列标注 · 第三章 NER：泛信息抽取能力 > 命名实体识别（NER）不仅仅局限于人名、地名、机构名的识别，其本质是一个通用的 span 分类器， > 3.1 传统认知的局限 > NER 的本质是通用 span 分类器，与具体实体类型无关。 · 第17页
[4] 《NLP序列标注任务详解》 · 序列标注 · 第二章 NLP 中的典型应用 > 2.1 中文分词（Tokenization） > 标签 > 标注示例 1（科技新闻）： · 第13页
```

