"""
第十周作业：上市公司年报 RAG 问答系统

基于 FAISS 向量检索 + DashScope LLM 的 RAG 问答系统。
数据来源：巨潮资讯网 5 家上市公司（茅台、五粮液、中国平安、宁德时代、海康威视）× 3 年 = 15 份年报。

用法：
  python qa_system.py                                    # 交互式问答
  python qa_system.py --query "贵州茅台2023年营业收入"     # 单次查询
  python qa_system.py --batch                            # 批量测试（questions.json 全部 20 题）
"""

import os
import json
import argparse
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────

# 索引路径（支持环境变量覆盖，解决中文路径兼容问题）
VECTORSTORE_DIR = Path(os.getenv(
    "RAG_VECTORSTORE_DIR",
    str(Path(__file__).parent / "vectorstore")
))
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH  = VECTORSTORE_DIR / "faiss_meta.json"

# DashScope API
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL   = "text-embedding-v3"
EMBED_DIM     = 1024
LLM_MODEL     = "qwen-plus"

# 检索参数
TOP_K          = 4     # 最终送入 LLM 的 chunk 数
SCORE_THRESHOLD = 0.25  # 相关性阈值，低于此值拒绝回答

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的财务分析助手，专门回答关于中国上市公司年度报告的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营业收入为1476亿元[1]
4. 数字要精确，不要四舍五入或模糊表达
5. 回答简洁，重点突出"""


# ── 核心类 ────────────────────────────────────────────────────────────────────

class QASystem:
    """基于 FAISS 向量检索的 RAG 问答系统。"""

    def __init__(self):
        # 初始化 DashScope 客户端
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)

        # 加载 FAISS 索引
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        print(f"[系统初始化] FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def _embed(self, text: str) -> np.ndarray:
        """调用 DashScope API 将文本转为归一化向量。"""
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[text], dimensions=EMBED_DIM
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def _format_source(self, meta: dict) -> str:
        """将 chunk 元数据格式化为可读来源。"""
        s = f"{meta.get('stock_code', '')} {meta.get('year', '')}年报"
        section = meta.get("section", "")
        if section:
            parts = section.split(" > ")
            s += " · " + " > ".join(parts[-2:])
        page = meta.get("page_num", -1)
        if page and page != -1:
            s += f" · 第{page}页"
        return s

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """向量检索：返回 top_k 个最相关的 chunk。"""
        q_vec = self._embed(query)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.meta_list[idx]
            results.append({
                "content": meta["content"],
                "source": self._format_source(meta),
                "chunk_id": meta.get("chunk_id", ""),
                "score": float(score),
            })
        return results

    def generate_answer(self, query: str, contexts: list) -> str:
        """调用 LLM 基于检索上下文生成回答。"""
        # 拼接参考资料
        ref_text = ""
        for i, ctx in enumerate(contexts, 1):
            ref_text += f"[{i}] {ctx['source']}\n{ctx['content']}\n\n"

        user_msg = f"【参考资料】\n{ref_text}【问题】\n{query}"

        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

    def ask(self, query: str, verbose: bool = True) -> dict:
        """完整 RAG 问答流程：检索 → 过滤 → 生成。"""
        # 1. 向量检索
        results = self.retrieve(query)

        # 2. 相关性检查
        if not results or results[0]["score"] < SCORE_THRESHOLD:
            answer = "根据年报知识库未能找到与该问题相关的内容。"
            return {"answer": answer, "contexts": [], "citations": []}

        # 3. LLM 生成
        answer = self.generate_answer(query, results)

        # 4. 组装引用
        citations = [{"index": i + 1, "source": r["source"]}
                     for i, r in enumerate(results)]

        if verbose:
            print(f"\n{'='*60}")
            print(f"问题：{query}")
            print(f"{'='*60}")
            print(f"\n{answer}\n")
            print("── 来源 ──")
            for c in citations:
                print(f"  [{c['index']}] {c['source']}")
            print()

        return {"answer": answer, "contexts": results, "citations": citations}


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="上市公司年报 RAG 问答系统")
    parser.add_argument("--query", type=str, help="单次查询（不指定则进入交互模式）")
    parser.add_argument("--batch", action="store_true", help="批量测试 questions.json 全部 20 题")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索返回的 chunk 数")
    args = parser.parse_args()

    qa = QASystem()

    if args.batch:
        # 批量测试模式
        q_path = Path(__file__).parent / "evaluation" / "questions.json"
        with open(q_path, encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        print(f"\n批量测试：共 {len(questions)} 道题\n")
        for q in questions:
            print(f"[Q{q['id']}] ({q['type']}) {q['question']}")
            result = qa.ask(q["question"], verbose=False)
            print(f"  → {result['answer'][:100]}...")
            print(f"  参考来源: {[c['source'] for c in result['citations'][:2]]}")
            print()

    elif args.query:
        # 单次查询模式
        qa.ask(args.query)

    else:
        # 交互式模式
        print("上市公司年报 RAG 问答系统（输入 exit 退出）")
        print(f"索引：{INDEX_PATH}（{qa.index.ntotal} 条向量）")
        print(f"模型：{LLM_MODEL}  |  Top-K：{args.top_k}")
        print("-" * 40)
        while True:
            try:
                query = input("\n请输入问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() == "exit":
                break
            qa.ask(query, verbose=True)


if __name__ == "__main__":
    main()
