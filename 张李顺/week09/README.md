# 比亚迪 2023 年报问答

基于 LangChain 的精简 RAG 实验。数据为[巨潮资讯网公开披露的比亚迪 2023 年年度报告](https://static.cninfo.com.cn/finalpage/2024-03-27/1219412018.PDF)，与教学数据公司不重叠。

## 启动

```powershell
conda activate ai_learning_1
cd my_project
$env:DASHSCOPE_API_KEY="sk-你的Key"
pip install -r requirements.txt
python download_data.py
python ingest.py
python evaluate.py
streamlit run app.py
```

`evaluate.py` 仅执行检索对比，按 10 个问题的人工标注相关页计算 `Recall@4`、`MRR@4` 与耗时，输出 `results/comparison.csv`、`results/query_details.csv` 和 `results/comparison.png`；不调用 LLM 评分或生成，控制时间与费用。网页采用递归分块 + 向量/BM25 混合检索，回答模型为 `qwen-plus`，向量模型为 `text-embedding-v3`。

Windows 版 FAISS 对中文路径写入不稳定，因此向量索引默认写入系统临时目录的 `byd_rag_vectorstore`。如需指定其他**纯英文路径**，启动前设置 `RAG_VECTORSTORE_DIR`。

数据、PDF、向量索引、结果文件及密钥都由 `.gitignore` 排除，不会推送到 Git。
