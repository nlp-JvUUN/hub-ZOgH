import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def seed_all(seed=None):
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(dataset, split):
    path = ROOT / CFG["datasets"][dataset] / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_rows(rows):
    return [x for x in rows if x["sentence1"] and x["sentence2"] and not any(c in x[k] for k in ("sentence1", "sentence2") for c in "\t\n\r")]


def balanced(rows, size, seed=42):
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for row in rows:
        groups[int(row["label"])].append(row)
    take = size // 2
    out = rng.sample(groups[0], min(take, len(groups[0]))) + rng.sample(groups[1], min(size - take, len(groups[1])))
    rng.shuffle(out)
    return out


def resolve_model(model_id):
    overrides = {
        CFG["models"]["encoder"]: os.getenv("ENCODER_MODEL"),
        CFG["models"]["llm"]: os.getenv("LLM_MODEL")
    }
    if overrides.get(model_id):
        return overrides[model_id]
    home = Path.home()
    if model_id == CFG["models"]["encoder"]:
        base = home / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"
        hits = sorted(base.glob("*")) if base.exists() else []
        if hits:
            return str(hits[-1])
    if model_id == CFG["models"]["llm"]:
        base = home / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen2-0___5B-Instruct"
        if base.exists():
            return str(base)
    return model_id


def output_dir(dataset, name=None):
    path = ROOT / "outputs" / dataset
    if name:
        path /= name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]
