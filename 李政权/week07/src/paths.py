"""项目路径常量。"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
BERT_PATH = Path(r"D:\basic-model\Bert\bert-base-chinese")
QWEN_PATH = Path(r"D:\basic-model\Qwen\Qwen2-1.5B")
