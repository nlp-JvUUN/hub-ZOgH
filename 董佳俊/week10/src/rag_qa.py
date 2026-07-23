"""
RAG 问答流水线 —— 英雄联盟知识库 (DeepSeek 版)。

流程: TF-IDF 向量化 → cosine 检索 top-k → DeepSeek LLM 生成回答

使用方式:
  python rag_qa.py                          # 交互式问答
  python rag_qa.py --query "亚索的技能是什么"  # 单次查询
  python rag_qa.py --query "..." --verbose    # 显示检索过程

要求环境变量: DEEPSEEK_API_KEY
"""
import json
import os
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np

BASE_DIR = Path(__file__).parent.parent
VEC_DIR = BASE_DIR / "vectorstore"
VEC_PATH = VEC_DIR / "vectors.npy"
VOCAB_PATH = VEC_DIR / "vocab.json"
META_PATH = VEC_DIR / "meta.json"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
LLM_MODEL = "deepseek-chat"

TOP_K = 4
SCORE_THRESHOLD = 0.15

SYSTEM_PROMPT = """你是英雄联盟知识问答助手。请根据【参考资料】回答用户问题。

回答规则:
1. 只根据参考资料中的内容回答，不编造数据
2. 若参考资料不足以支撑回答，说"根据手头资料无法回答此问题"
3. 引用内容时标注来源编号，如 [1][2]
4. 回答简洁准确，不展开无关话题"""


def get_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    return key


def tokenize(text: str) -> list[str]:
    chars = list(text)
    tokens = []
    for c in chars:
        if c.strip():
            tokens.append(c)
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if bigram.strip():
            tokens.append(bigram)
    return tokens


class RAGPipeline:
    def __init__(self):
        self.api_key = get_api_key()
        self.vectors = np.load(VEC_PATH).astype("float32")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.term_to_idx = {t: i for i, t in enumerate(self.vocab)}
        print(f"索引加载完成: {self.vectors.shape[0]} 条向量 x {self.vectors.shape[1]} 维")

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        counter = Counter(tokens)
        max_freq = max(counter.values()) if counter else 1
        vec = np.zeros(len(self.vocab), dtype="float32")
        for term, count in counter.items():
            if term in self.term_to_idx:
                vec[self.term_to_idx[term]] = count / max_freq
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-9)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        query_vec = self._vectorize(query)
        scores = self.vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = dict(self.meta[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results

    def _call_llm(self, query: str, contexts: list[dict]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            title = ctx["metadata"]["title"]
            content = ctx["content"]
            parts.append(f"[{i}] 来源: {title}\n{content}")

        context_text = "\n\n---\n\n".join(parts)
        user_msg = (
            f"【参考资料】\n{context_text}\n\n"
            f"【问题】\n{query}\n\n"
            "请根据参考资料回答，引用处标注来源编号。"
        )

        payload = json.dumps({
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
        }).encode("utf-8")

        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def query(self, question: str, verbose: bool = False) -> dict:
        results = self.search(question, TOP_K)

        if verbose:
            print(f"\n[检索结果] top-{TOP_K}:")
            for i, r in enumerate(results, 1):
                title = r["metadata"]["title"]
                score = r["score"]
                preview = r["content"][:80].replace("\n", " ")
                print(f"  {i}. [{score:.3f}] {title}: {preview}...")

        if not results or results[0]["score"] < SCORE_THRESHOLD:
            return {
                "answer": "根据手头资料暂时无法回答这个问题。",
                "sources": [],
            }

        answer = self._call_llm(question, results)

        sources = [{"index": i+1, "source": r["metadata"]["title"],
                     "score": round(r["score"], 3)} for i, r in enumerate(results)]

        return {"answer": answer, "sources": sources}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="英雄联盟 RAG 问答 (DeepSeek)")
    parser.add_argument("--query", type=str, default=None, help="单次查询")
    parser.add_argument("--verbose", action="store_true", help="显示检索过程")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.query:
        result = pipeline.query(args.query, verbose=args.verbose)
        print(f"\n{'='*60}")
        print(f"问: {args.query}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["sources"]:
            print("\n── 来源 ──")
            for s in result["sources"]:
                print(f"  [{s['index']}] {s['source']} (score={s['score']})")
    else:
        print(f"\n英雄联盟 RAG 问答系统 (DeepSeek)")
        print(f"模型: {LLM_MODEL}  |  向量: {VEC_PATH.name}")
        print("输入问题开始，'exit' 退出\n")

        while True:
            try:
                q = input("问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break

            result = pipeline.query(q, verbose=True)
            print(f"\n{result['answer']}")
            if result["sources"]:
                print("\n── 来源 ──")
                for s in result["sources"]:
                    print(f"  [{s['index']}] {s['source']}")


if __name__ == "__main__":
    main()
