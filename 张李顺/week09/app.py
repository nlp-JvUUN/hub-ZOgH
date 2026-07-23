import pickle
import time
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import API_KEY, BASE_URL, VECTOR_DIR, EMBED_MODEL, CHAT_MODEL

st.set_page_config(page_title="比亚迪年报问答", page_icon="🚗", layout="wide")


@st.cache_resource
def embedder():
    return OpenAIEmbeddings(
        model=EMBED_MODEL, api_key=API_KEY, base_url=BASE_URL, dimensions=1024,
        check_embedding_ctx_length=False, chunk_size=10,
    )


@st.cache_resource
def load_resources():
    e = embedder()
    return FAISS.load_local(str(VECTOR_DIR / "recursive"), e, allow_dangerous_deserialization=True), pickle.load(open(VECTOR_DIR / "bm25.pkl", "rb"))


def hybrid_search(query, store, bm25_data, k=5):
    dense = store.similarity_search(query, k=k)
    scores = bm25_data["bm25"].get_scores(list(re.sub(r"\s+", "", query)))
    lexical = [bm25_data["chunks"][i] for i in scores.argsort()[-k:][::-1]]
    merged, seen = [], set()
    for doc in dense + lexical:
        key = (doc.metadata["page"], doc.page_content[:80])
        if key not in seen:
            merged.append(doc); seen.add(key)
    return merged[:k]


st.title("🚗 比亚迪 2023 年报问答")
st.caption("LangChain · DashScope qwen-plus · text-embedding-v3 · 来源：巨潮资讯网公开年报")
if not API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY。请在启动终端设置后重新运行。")
    st.stop()
if not (VECTOR_DIR / "recursive").exists():
    st.warning("尚未建立本地索引，请先运行：python ingest.py")
    st.stop()

question = st.text_input("请输入问题", placeholder="例如：比亚迪 2023 年的主要业务有哪些？")
if st.button("开始问答", type="primary") and question:
    store, bm25_data = load_resources()
    start = time.perf_counter()
    docs = hybrid_search(question, store, bm25_data)
    context = "\n\n".join(f"[第{d.metadata['page']}页] {d.page_content}" for d in docs)
    prompt = f"你是年度报告助手。仅根据下列资料回答，缺少依据时明确说明。答案简洁，并在相应句子后标注[第X页]。\n\n资料：\n{context}\n\n问题：{question}"
    answer = ChatOpenAI(model=CHAT_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.1).invoke(prompt).content
    st.subheader("回答")
    st.write(answer)
    st.caption(f"检索与生成耗时：{time.perf_counter() - start:.2f} 秒")
    with st.expander("查看引用原文"):
        for doc in docs:
            st.markdown(f"**第 {doc.metadata['page']} 页**")
            st.write(doc.page_content)
