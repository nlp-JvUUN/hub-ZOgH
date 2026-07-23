"""
BERT 文本分类训练器 —— 支持三种微调策略的对比实验

三种策略（通过 --method 指定）：
  full   : 全量微调 —  BERT 全部参数参与训练（参数量最大，效果通常最好）
  freeze : 冻结微调 —  只用 BERT 提取特征，不更新 BERT 参数（最快，效果最差）
  lora   : LoRA 微调 —  低秩适配器（参数量 ~0.3%，速度和效果折中）

三种策略通过 --method 参数切换，可直接对比训练效果。

用法：
  python src_bert/bert_trainer.py --method full     # 全量微调
  python src_bert/bert_trainer.py --method freeze   # 冻结 BERT
  python src_bert/bert_trainer.py --method lora     # LoRA 微调
  python src_bert/bert_trainer.py --epochs 1 --batch_size 16 --dry  # 快速验证
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
MODEL_CACHE = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


# ── 数据集封装 ──

class ClsDataset(Dataset):
    """文本分类 Dataset：text → tokenize → {input_ids, attention_mask, label}。"""

    def __init__(self, records: list[dict], tokenizer, max_len: int = 256):
        self.texts = [r["text"] for r in records]
        self.labels = [r["label"] for r in records]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── 评估 ──

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """在 loader 上评估，返回 accuracy / precision / recall / f1。"""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, _ = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 6),
        "precision": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 6),
        "recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 6),
        "f1": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 6),
    }


# ── 训练一个 epoch ──

def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer,
        scheduler,
        device: torch.device,
        grad_accum: int,
        epoch: int,
        total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        _, loss = model(input_ids, attention_mask, labels=labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 处理尾部
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


# ── 主流程 ──

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # —— 数据 ——
    import sys; sys.path.insert(0, str(ROOT))
    from data_loader import load_dataset, get_num_classes

    train_records = load_dataset("train")
    val_records = load_dataset("validation")
    test_records = load_dataset("test")
    num_classes = get_num_classes()
    print(f"数据: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    if args.dry:
        train_records = train_records[:50]
        val_records = val_records[:50]

    # —— Tokenizer ——
    model_path = str(args.bert_path) if args.bert_path else "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # —— DataLoader ——
    def _make_loader(records, shuffle=False):
        ds = ClsDataset(records, tokenizer, max_len=args.max_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(train_records, shuffle=True)
    val_loader = _make_loader(val_records)
    test_loader = _make_loader(test_records)

    # —— 模型 ——
    from bert_model import build_classifier

    model = build_classifier(
        method=args.method,
        model_name=model_path,
        num_classes=num_classes,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # —— 优化器 & 调度器 ——
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"  总步数: {total_steps}, warmup: {warmup}")

    # —— 训练循环 ——
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_cls_{args.method}.pt"
    best_f1 = 0.0
    log_records = []

    print(f"\n{'=' * 60}")
    print(f"  开始训练（{args.method} 模式）")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device,
                                 args.grad_accum, epoch, args.epochs)

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_s": round(elapsed, 1),
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "method": args.method,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"    ★ 最优 checkpoint (F1={best_f1:.4f}) → {ckpt_path}")

    # —— 测试集评估 ——
    print(f"\n{'=' * 60}")
    print(f"  测试集评估")
    print(f"{'=' * 60}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  F1-macro : {test_metrics['f1']:.4f}")

    # —— 保存日志 ——
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": f"BERT-{args.method}",
        "model_name": model_path,
        "test_metrics": test_metrics,
        "training_log": log_records,
        "best_val_f1": best_f1,
    }
    log_path = LOG_DIR / f"bert_{args.method}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  日志已保存 → {log_path}")
    print(f"  下一步：python src_llm/prompt_classify.py  # LLM 对比实验")


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类 — 三种微调策略对比")
    parser.add_argument("--method", choices=["full", "freeze", "lora"], default="full",
                        help="微调策略（默认 full）")
    parser.add_argument("--bert_path", type=Path, default=None,
                        help="本地 BERT 模型路径（默认从 HuggingFace 下载 bert-base-chinese）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--dry", action="store_true", help="小数据快速验证")
    return parser.parse_args()


if __name__ == "__main__":
    main()
