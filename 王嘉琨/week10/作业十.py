import os
import json
import argparse
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI


# ── 配置 ─────────────────────────────────────────────────────────────────────

# 向量数据库路径（支持环境变量覆盖，解决中文路径兼容问题）
VECTORSTORE_DIR = Path(os.getenv(
    "RAG_VECTORSTORE_DIR",
    str(Path(__file__).parent / "vectorstore")
))

# OpenAI API 相关配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("请设置环境变量 OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text - embedding - ada - 002"
LLM_MODEL = "gpt - 3.5 - turbo"

# 检索参数
TOP_K = 3
SCORE_THRESHOLD = 0.7

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的学术知识助手，专门回答关于学术论文的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的信息
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体信息时，在句末标注来源编号，如：该结论已在多项研究中被证实[1]
4. 回答简洁，重点突出"""


# ── 核心类 ────────────────────────────────────────────────────────────────────

class QASystem:
    """基于 Chromadb 向量检索的 RAG 问答系统。"""

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="academic_papers",
            embedding_function=OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
        )
        print(f"[系统初始化] Chromadb 向量数据库加载完成，共 {self.collection.count()} 条向量")

    def _format_source(self, doc):
        """将文档元数据格式化为可读来源。"""
        source = doc.get("source", "未知来源")
        page = doc.get("page", -1)
        if page and page != -1:
            source += f" · 第{page}页"
        return source

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """向量检索：返回 top_k 个最相关的文档。"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        contexts = []
        for i in range(len(results['documents'][0])):
            score = 1 - results['distances'][0][i]
            if score < SCORE_THRESHOLD:
                continue
            doc = {
                "content": results['documents'][0][i],
                "source": self._format_source(results['metadatas'][0][i]),
                "score": score
            }
            contexts.append(doc)
        return contexts

    def generate_answer(self, query: str, contexts: list) -> str:
        """调用 LLM 基于检索上下文生成回答。"""
        # 拼接参考资料
        ref_text = ""
        for i, ctx in enumerate(contexts, 1):
            ref_text += f"[{i}] {ctx['source']}\n{ctx['content']}\n\n"

        user_msg = f"【参考资料】\n{ref_text}【问题】\n{query}"

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )
        return resp.choices[0].message.content

    def ask(self, query: str, verbose: bool = True) -> dict:
        """完整 RAG 问答流程：检索 → 过滤 → 生成。"""
        # 1. 向量检索
        results = self.retrieve(query)

        # 2. 相关性检查
        if not results:
            answer = "根据学术论文知识库未能找到与该问题相关的内容。"
            return {"answer": answer, "contexts": [], "citations": []}

        # 3. LLM 生成
        answer = self.generate_answer(query, results)

        # 4. 组装引用
        citations = [{"index": i + 1, "source": r["source"]}
                     for i, r in enumerate(results)]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"问题：{query}")
            print(f"{'=' * 60}")
            print(f"\n{answer}\n")
            print("── 来源 ──")
            for c in citations:
                print(f"  [{c['index']}] {c['source']}")
            print()

        return {"answer": answer, "contexts": results, "citations": citations}


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="学术论文 RAG 问答系统")
    parser.add_argument("--query", type=str, help="单次查询（不指定则进入交互模式）")
    parser.add_argument("--batch", action="store_true", help="批量测试 questions.json 全部问题")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索返回的文档数")
    args = parser.parse_args()

    qa = QASystem()

    if args.batch:
        # 批量测试模式
        q_path = Path(__file__).parent / "evaluation" / "questions.json"
        with open(q_path, encoding="utf - 8") as f:
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
        print("学术论文 RAG 问答系统（输入 exit 退出）")
        print(f"索引：学术论文向量库（{qa.collection.count()} 条向量）")
        print(f"模型：{LLM_MODEL}  |  Top - K：{args.top_k}")
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

