from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import CFG, chunks, device, resolve_model


SYSTEM = "判断两个中文句子的语义是否相同。只回答：是或否。"


def prompt(row, tok):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"句子一：{row['sentence1']}\n句子二：{row['sentence2']}\n语义是否相同？"}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_base():
    path = resolve_model(CFG["models"]["llm"])
    local = Path(path).exists()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=local)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=local, torch_dtype=torch.float16 if device().type == "cuda" else torch.float32, attn_implementation="sdpa")
    return model.to(device()), tok


@torch.inference_mode()
def scores(model, tok, rows, batch_size=16):
    model.eval()
    tok.padding_side = "left"
    yes = tok.encode("是", add_special_tokens=False)[0]
    no = tok.encode("否", add_special_tokens=False)[0]
    out = []
    for batch in chunks(rows, batch_size):
        inputs = tok([prompt(x, tok) for x in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device()) for k, v in inputs.items()}
        logits = model(**inputs).logits[:, -1].float()
        out.extend((logits[:, yes] - logits[:, no]).cpu().tolist())
    return np.array(out)


def attach_adapter(model, path):
    return PeftModel.from_pretrained(model, path).to(device())
