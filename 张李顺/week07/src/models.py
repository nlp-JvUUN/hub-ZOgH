import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .common import CFG, chunks, device, resolve_model


class BiEncoder(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path or resolve_model(CFG["models"]["encoder"])
        self.backbone = AutoModel.from_pretrained(self.model_path, local_files_only=Path(self.model_path).exists())

    def forward(self, batch):
        return F.normalize(self.backbone(**batch).last_hidden_state[:, 0], dim=-1)


class CrossEncoder(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path or resolve_model(CFG["models"]["encoder"])
        self.backbone = AutoModel.from_pretrained(self.model_path, local_files_only=Path(self.model_path).exists())
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, batch):
        return self.head(self.backbone(**batch).last_hidden_state[:, 0]).squeeze(-1)


def tokenizer():
    path = resolve_model(CFG["models"]["encoder"])
    return AutoTokenizer.from_pretrained(path, local_files_only=Path(path).exists())


def tokenize(tok, left, right=None, max_length=None):
    data = tok(left, right, padding=True, truncation=True, max_length=max_length or CFG["train"]["max_length"], return_tensors="pt")
    return {k: v.to(device()) for k, v in data.items()}


@torch.inference_mode()
def encode(model, tok, texts, batch_size=256):
    model.eval()
    out = []
    for batch in chunks(texts, batch_size):
        out.append(model(tokenize(tok, batch)).float().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def cross_scores(model, tok, rows, batch_size=256):
    model.eval()
    out = []
    for batch in chunks(rows, batch_size):
        left = [x["sentence1"] for x in batch]
        right = [x["sentence2"] for x in batch]
        out.append(model(tokenize(tok, left, right)).float().cpu().numpy())
    return np.concatenate(out)


def save_model(model, path, meta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bi(path):
    model = BiEncoder()
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device())


def load_cross(path):
    model = CrossEncoder()
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device())
