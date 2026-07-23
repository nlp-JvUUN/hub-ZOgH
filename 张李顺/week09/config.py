from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "BYD_2023_annual_report.pdf"
PAGES_PATH = DATA_DIR / "pages.json"
# Windows FAISS wheels cannot reliably write to paths containing Chinese
# characters.  Use an ASCII temp path by default; callers may override it.
VECTOR_DIR = Path(os.getenv("RAG_VECTORSTORE_DIR", str(Path(tempfile.gettempdir()) / "byd_rag_vectorstore")))
RESULTS_DIR = ROOT / "results"
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
CHAT_MODEL = "qwen-plus"
PDF_URL = "https://static.cninfo.com.cn/finalpage/2024-03-27/1219412018.PDF"
SOURCE_NAME = "比亚迪股份有限公司 2023 年年度报告"
