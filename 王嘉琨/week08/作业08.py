import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_crossencoder_loaders
from evaluate import eval_crossencoder, eval_biencoder
from model import build_crossencoder, build_biencoder

from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import torch


def build_biencoder_loaders(data_dir, tokenizer, max_length, batch_size):
    # 加载数据的逻辑，这里假设数据存储在一个 JSON 文件中，格式为 [{"s1": "句子1", "s2": "句子2", "label": 0/1}]
    with open(data_dir / 'data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_ids1_list, attention_mask1_list, input_ids2_list, attention_mask2_list, labels_list = [], [], [], [], []
    for item in data:
        s1 = item['s1']
        s2 = item['s2']
        label = item['label']

        encoding1 = tokenizer(s1, truncation=True, padding='max_length', max_length=max_length)
        encoding2 = tokenizer(s2, truncation=True, padding='max_length', max_length=max_length)

        input_ids1_list.append(torch.tensor(encoding1['input_ids']))
        attention_mask1_list.append(torch.tensor(encoding1['attention_mask']))
        input_ids2_list.append(torch.tensor(encoding2['input_ids']))
        attention_mask2_list.append(torch.tensor(encoding2['attention_mask']))
        labels_list.append(torch.tensor(label))

    input_ids1_tensor = torch.stack(input_ids1_list)
    attention_mask1_tensor = torch.stack(attention_mask1_list)
    input_ids2_tensor = torch.stack(input_ids2_list)
    attention_mask2_tensor = torch.stack(attention_mask2_list)
    labels_tensor = torch.stack(labels_list)

    dataset = TensorDataset(input_ids1_tensor, attention_mask1_tensor, input_ids2_tensor, attention_mask2_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, None, None  # 返回训练集加载器，这里省略验证集和测试集加载器的构建
# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert - base - chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Model]", leave=False)
    for step, batch in enumerate(pbar):
        if model.__class__.__name__ == 'CrossEncoder':
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
        else:
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model_types = ['crossencoder', 'biencoder']
    for model_type in model_types:
        print(f"\n开始训练 {model_type} 模型...")
        if model_type == 'crossencoder':
            tokenizer = BertTokenizer.from_pretrained(args.bert_path)
            print("\nDataLoader 构建中...")
            train_loader, val_loader, _ = build_crossencoder_loaders(
                args.data_dir, tokenizer,
                max_length=args.max_length, batch_size=args.batch_size,
            )
            model = build_crossencoder(
                bert_path=args.bert_path,
                num_hidden_layers=args.num_hidden_layers,
            ).to(device)
            eval_func = eval_crossencoder
        else:
            tokenizer = BertTokenizer.from_pretrained(args.bert_path)
            print("\nDataLoader 构建中...")
            train_loader, val_loader, _ = build_biencoder_loaders(
                args.data_dir, tokenizer,
                max_length=args.max_length, batch_size=args.batch_size,
            )
            model = build_biencoder(
                bert_path=args.bert_path,
                num_hidden_layers=args.num_hidden_layers,
            ).to(device)
            eval_func = eval_biencoder

        # ── 分层学习率 ────────────────────────────────────────────────────────
        if hasattr(model, 'bert'):
            bert_params = list(model.bert.parameters())
            head_params = (list(model.dropout.parameters()) +
                           list(model.classifier.parameters()))
        else:
            bert_params = list(model.bert1.parameters()) + list(model.bert2.parameters())
            head_params = list(model.fc.parameters())

        optimizer = AdamW([
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ], weight_decay=0.01)

        total_steps = len(train_loader) * args.epochs // args.grad_accum
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")

        criterion = nn.CrossEntropyLoss()

        # ── 训练循环 ──────────────────────────────────────────────────────────
        ckpt_path = CKPT_DIR / f"{model_type}_best.pt"
        best_val_f1 = 0.0
        log_records = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, epoch, args.epochs, args.grad_accum,
            )

            val_metrics = eval_func(model, val_loader, device)
            elapsed = time.time() - t0

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1"]
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                  f"{elapsed:.0f}s")

            log_records.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "val_f1": val_f1, "elapsed_s": elapsed,
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

        # ── 训练完成，保存日志 ────────────────────────────────────────────────
        log_path = LOG_DIR / f"{model_type}_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_records, f, ensure_ascii=False, indent=2)
        print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
        print(f"训练日志 → {log_path}")
        print(f"最优 checkpoint → {ckpt_path}")


# ── 参数解析 ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="文本匹配模型训练")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--num_hidden_layers", default=4, type=int,
                        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int,
                        help="句对总最大 token 数（两句拼接，建议 128）")
    parser.add_argument("--lr", default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", default=5.0, type=float, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
