import argparse
import time

import torch
from torch import nn

from .common import CFG, balanced, clean_rows, device, load_rows, output_dir, seed_all
from .mining import mine_triplets
from .models import BiEncoder, save_model, tokenize, tokenizer


def train(dataset, loss_name, max_seconds=None):
    seed_all()
    cfg = CFG["train"]
    max_seconds = max_seconds or cfg["max_train_seconds"]
    rows = clean_rows(load_rows(dataset, "train"))
    model = BiEncoder().to(device())
    tok = tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device().type == "cuda")
    cosine_loss = nn.CosineEmbeddingLoss(margin=.3)
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y), margin=.2)
    if loss_name == "cosine":
        samples = balanced(rows, cfg["bi_train_size"])
        batch_size = cfg["bi_batch_size"]
    else:
        samples = mine_triplets(rows, cfg["bi_train_size"], CFG["seed"])
        batch_size = cfg["triplet_batch_size"]
    started = time.monotonic()
    step = 0
    losses = []
    model.train()
    while step < cfg["bi_max_steps"] and time.monotonic() - started < max_seconds:
        for i in range(0, len(samples), batch_size):
            if step >= cfg["bi_max_steps"] or time.monotonic() - started >= max_seconds:
                break
            batch = samples[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device().type == "cuda"):
                if loss_name == "cosine":
                    a = model(tokenize(tok, [x["sentence1"] for x in batch]))
                    b = model(tokenize(tok, [x["sentence2"] for x in batch]))
                    y = torch.tensor([1 if x["label"] else -1 for x in batch], device=device(), dtype=a.dtype)
                    loss = cosine_loss(a, b, y)
                else:
                    a = model(tokenize(tok, [x[0] for x in batch]))
                    p = model(tokenize(tok, [x[1] for x in batch]))
                    n = model(tokenize(tok, [x[2] for x in batch]))
                    loss = triplet_loss(a, p, n)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
    elapsed = time.monotonic() - started
    target = output_dir(dataset, f"bi_{loss_name}")
    save_model(model, target, {"dataset": dataset, "loss": loss_name, "steps": step, "seconds": elapsed, "mean_loss": sum(losses) / len(losses)})
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CFG["datasets"], required=True)
    parser.add_argument("--loss", choices=["cosine", "triplet"], required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    train(args.dataset, args.loss, args.max_seconds)


if __name__ == "__main__":
    main()
