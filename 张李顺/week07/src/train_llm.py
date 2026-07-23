import argparse
import json
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model

from .common import CFG, balanced, clean_rows, device, load_rows, output_dir, seed_all
from .llm import load_base, prompt


def encode_rows(rows, tok):
    data = []
    tok.padding_side = "right"
    for row in rows:
        prefix = prompt(row, tok)
        answer = "是" if row["label"] else "否"
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        ids = tok(prefix + answer + tok.eos_token, add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
        labels = [-100] * min(len(prefix_ids), len(ids)) + ids[len(prefix_ids):]
        data.append((ids, labels))
    return data


def collate(batch, pad):
    width = max(len(x[0]) for x in batch)
    ids, masks, labels = [], [], []
    for x, y in batch:
        n = width - len(x)
        ids.append(x + [pad] * n)
        masks.append([1] * len(x) + [0] * n)
        labels.append(y + [-100] * n)
    return torch.tensor(ids, device=device()), torch.tensor(masks, device=device()), torch.tensor(labels, device=device())


def train(dataset, max_seconds=None):
    seed_all()
    cfg = CFG["train"]
    max_seconds = max_seconds or cfg["max_train_seconds"]
    rows = balanced(clean_rows(load_rows(dataset, "train")), cfg["llm_train_size"])
    model, tok = load_base()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    data = encode_rows(rows, tok)
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device().type == "cuda")
    started = time.monotonic()
    losses = []
    step = 0
    model.train()
    while step < cfg["llm_max_steps"] and time.monotonic() - started < max_seconds:
        for i in range(0, len(data), cfg["llm_batch_size"]):
            if step >= cfg["llm_max_steps"] or time.monotonic() - started >= max_seconds:
                break
            ids, masks, labels = collate(data[i:i + cfg["llm_batch_size"]], tok.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device().type == "cuda"):
                loss = model(input_ids=ids, attention_mask=masks, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((x for x in model.parameters() if x.requires_grad), 1)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
    elapsed = time.monotonic() - started
    target = output_dir(dataset, "llm_lora")
    model.save_pretrained(target)
    (target / "meta.json").write_text(json.dumps({"dataset": dataset, "steps": step, "seconds": elapsed, "mean_loss": sum(losses) / len(losses)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CFG["datasets"], required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    train(args.dataset, args.max_seconds)


if __name__ == "__main__":
    main()
