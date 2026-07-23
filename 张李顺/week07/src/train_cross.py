import argparse
import time

import torch
from torch import nn

from .common import CFG, balanced, clean_rows, device, load_rows, output_dir, seed_all
from .models import CrossEncoder, save_model, tokenize, tokenizer


def train(dataset, max_seconds=None):
    seed_all()
    cfg = CFG["train"]
    max_seconds = max_seconds or cfg["max_train_seconds"]
    samples = balanced(clean_rows(load_rows(dataset, "train")), cfg["cross_train_size"])
    model = CrossEncoder().to(device())
    tok = tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device().type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    started = time.monotonic()
    losses = []
    step = 0
    model.train()
    while step < cfg["cross_max_steps"] and time.monotonic() - started < max_seconds:
        for i in range(0, len(samples), cfg["cross_batch_size"]):
            if step >= cfg["cross_max_steps"] or time.monotonic() - started >= max_seconds:
                break
            batch = samples[i:i + cfg["cross_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device().type == "cuda"):
                logits = model(tokenize(tok, [x["sentence1"] for x in batch], [x["sentence2"] for x in batch]))
                labels = torch.tensor([x["label"] for x in batch], device=device(), dtype=logits.dtype)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
    elapsed = time.monotonic() - started
    target = output_dir(dataset, "cross")
    save_model(model, target, {"dataset": dataset, "loss": "binary_cross_entropy", "steps": step, "seconds": elapsed, "mean_loss": sum(losses) / len(losses)})
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CFG["datasets"], required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    train(args.dataset, args.max_seconds)


if __name__ == "__main__":
    main()
