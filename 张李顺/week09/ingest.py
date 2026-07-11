"""Extract pages, create two LangChain vector stores, and a BM25 index."""
import json
import pickle
import re
import fitz
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from config import API_KEY, BASE_URL, DATA_DIR, PDF_PATH, PAGES_PATH, VECTOR_DIR, EMBED_MODEL, SOURCE_NAME


def embeddings():
    if not API_KEY:
        raise EnvironmentError("Set DASHSCOPE_API_KEY before building the index.")
    return OpenAIEmbeddings(
        model=EMBED_MODEL, api_key=API_KEY, base_url=BASE_URL, dimensions=1024,
        check_embedding_ctx_length=False,
        # DashScope text-embedding-v3 accepts at most 10 inputs per request.
        chunk_size=10,
    )


def extract_pages():
    if not PDF_PATH.exists():
        raise FileNotFoundError("Run: python download_data.py")
    pages = []
    with fitz.open(PDF_PATH) as pdf:
        for number, page in enumerate(pdf, 1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            if len(text) >= 80:
                pages.append({"page": number, "text": text})
    DATA_DIR.mkdir(exist_ok=True)
    PAGES_PATH.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return pages


def documents(pages):
    return [Document(page_content=item["text"], metadata={"page": item["page"], "source": SOURCE_NAME}) for item in pages]


def build_store(name, docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "。", "；", "，", " "])
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings())
    target = VECTOR_DIR / name
    # FAISS.save_local writes files into this exact directory but does not
    # create it itself.
    target.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))
    return chunks


def main():
    pages = extract_pages()
    docs = documents(pages)
    VECTOR_DIR.mkdir(exist_ok=True)
    fixed_chunks = build_store("fixed", docs, 800, 0)
    recursive_chunks = build_store("recursive", docs, 800, 120)
    # Chinese annual reports have no whitespace word boundaries; character tokens
    # keep the lightweight BM25 baseline useful without an extra tokenizer.
    tokens = [list(re.sub(r"\s+", "", chunk.page_content)) for chunk in recursive_chunks]
    with open(VECTOR_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi(tokens), "chunks": recursive_chunks}, f)
    print(f"Indexed {len(pages)} pages; fixed={len(fixed_chunks)}, recursive={len(recursive_chunks)}")


if __name__ == "__main__":
    main()
