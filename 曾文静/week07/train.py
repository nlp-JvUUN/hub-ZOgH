# -*- coding: utf-8 -*-
"""
week07/train.py
===============
BERT 序列标注训练: BERT+Linear(基线) vs BERT+CRF

用法:
  python train.py                        # BERT + Linear 基线
  python train.py --use_crf              # BERT + CRF
  python train.py --epochs 5 --lr 3e-5 --batch_size 16
  python train.py --num_train 2000       # 快速实验(只用前2000条)

输出:
  outputs/checkpoints/best_{linear|crf}.pt   最优验证F1的模型
  outputs/figures/train_{linear|crf}.png     损失/验证F1曲线
  outputs/logs/train_{linear|crf}.json       训练日志
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import build_dataloaders, build_label_schema
from metrics import entity_f1
from model import NERModel


def predict_tags(model, loader, dataset, id2label, device):
    """
    遍历验证/测试集, 返回 (gold_tags, pred_tags):
      都是 list[list[str]], 每个样本是 token 级标签序列
    """
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            _, decoded = model(input_ids, attention_mask, token_type_ids)

            start = step * loader.batch_size
            for b in range(input_ids.size(0)):
                idx = start + b
                if idx >= len(dataset):
                    break
                wids = dataset.word_ids[idx]
                seq = decoded[b] if model.use_crf else decoded[b].tolist()

                tok_preds, prev_wid = [], None
                for j, wid in enumerate(wids):
                    if wid is None or wid == prev_wid:   # 特殊token/非首子词
                        continue
                    if j >= len(seq):                    # CRF 可能已裁剪padding
                        break
                    tok_preds.append(id2label[seq[j]])
                    prev_wid = wid
                # 截断对齐(与预测长度一致, 防止截断导致长度不匹配)
                gold.append(dataset.true_tags[idx][:len(tok_preds)])
                pred.append(tok_preds)
    return gold, pred


def plot_history(history, out_png, tag):
    """绘制训练损失和验证F1曲线"""
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["loss"], "o-", color="#4C72B0", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_f1"], "s-", color="#DD8452", label="val entity F1")
    ax2.set_ylabel("entity F1", color="#DD8452")
    ax2.set_ylim(0, 1)
    fig.suptitle(f"BERT + {'CRF' if tag == 'crf' else 'Linear'}")
    fig.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BERT 序列标注训练(cluener2020)")
    parser.add_argument("--use_crf", action="store_true", help="使用CRF解码头")
    parser.add_argument("--bert_path", type=str, default="bert-base-chinese")
    parser.add_argument("--data_dir", type=str, default="data/cluener")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0,
                        help="分类头/CRF层学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train", type=int, default=None,
                        help="限制训练样本数, 快速实验用")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tag = "crf" if args.use_crf else "linear"
    out_dir = Path(args.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device} | 模型: BERT+{tag.upper()}")

    # ---------- 数据 ----------
    labels, label2id, id2label = build_label_schema()
    print(f"[INFO] 标签体系: {len(labels)} 个标签 ({', '.join(labels[:5])} ...)")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=True)
    loaders = build_dataloaders(args.data_dir, tokenizer, label2id,
                                args.batch_size, args.max_length,
                                args.num_train)

    # ---------- 模型 ----------
    model = NERModel(args.bert_path, num_labels=len(labels),
                     use_crf=args.use_crf).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 可训练参数量: {n_params:,}")

    # 分层学习率: BERT 主体低学习率, 分类头/CRF 高学习率
    bert_params = [p for n, p in model.named_parameters() if "bert" in n]
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ])
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps)

    # ---------- 训练 ----------
    history = {"loss": [], "val_f1": []}
    best_f1 = 0.0
    ckpt_path = out_dir / "checkpoints" / f"best_{tag}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            # CRF 用完整标签, Linear 用 -100 掩码标签
            labels_b = (batch[4] if args.use_crf else batch[3]).to(device)
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * input_ids.size(0)
        train_loss = epoch_loss / len(loaders["train"].dataset)

        gold, pred = predict_tags(model, loaders["validation"],
                                  loaders["validation"].dataset, id2label, device)
        m = entity_f1(gold, pred)
        history["loss"].append(round(train_loss, 4))
        history["val_f1"].append(m["f1"])
        print(f"  epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | "
              f"val P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f} "
              f"| {time.time()-t0:.0f}s")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"state_dict": model.state_dict(), "use_crf": args.use_crf,
                        "label2id": label2id, "best_f1": best_f1}, ckpt_path)
            print(f"  ★ 最优验证F1 {best_f1:.4f}, checkpoint 已保存 -> {ckpt_path}")

    plot_history(history, out_dir / "figures" / f"train_{tag}.png", tag)
    with open(out_dir / "logs" / f"train_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_f1": best_f1}, f, ensure_ascii=False, indent=2)
    print(f"\n完成! 最优验证 entity F1 = {best_f1:.4f}")
    print(f"曲线图: {out_dir/'figures'}/train_{tag}.png")


if __name__ == "__main__":
    main()
