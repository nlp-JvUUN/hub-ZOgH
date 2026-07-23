"""
BERT NER 训练脚本

教学重点：
  1. --use_crf 参数：一套脚本同时支持两种模型
  2. 分层学习率：BERT 层用 2e-5，分类头用 1e-4（加速头部收敛）
  3. Linear Warmup：防止训练初期大梯度破坏预训练参数
  4. seqeval 评估：entity-level F1（不是 token-level accuracy）

使用方式：
  python train.py                        # 训练 BERT+Linear（基线）
  python train.py --use_crf              # 训练 BERT+CRF
  python train.py --epochs 5 --lr 3e-5  # 自定义超参数

依赖：
  pip install torch transformers seqeval pytorch-crf tqdm
  export DASHSCOPE_API_KEY="sk-xxx"   （LLM对比时用）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_label_schema, build_dataloaders
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。

    适配 dataset.py 的子词对齐格式：
      - labels 中 -100 表示不参与 loss 的位置（特殊 token、非首子词、PAD）
      - gold_seq 只收集 labels != -100 的位置（即首子词对应的真实标签）
      - pred_seq 同样只收集对应位置的预测标签
      - 统一使用 model.decode() 获取预测结果（BertNER 和 BertCRFNER 均已实现）
    """
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            # 将 batch 数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播，计算 loss
            _, loss = model(input_ids, attention_mask, token_type_ids, labels)

            # 统一使用 model.decode() 获取预测结果
            # BertNER.decode() 返回 list[list[int]]，按 attention_mask 截断 PAD
            # BertCRFNER.decode() 返回 list[list[int]]，Viterbi 解码结果
            pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

            total_loss += loss.item()

            # 将预测 id 和真实标签对齐，构建 seqeval 所需的标签序列
            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i]

                # 维护 pred_seq 的索引：因为 pred_ids 的长度可能与
                # token_labels 的有效位置数不同（CRF decode 不含 PAD），
                # 所以需要单独跟踪 pred_ids 中的位置
                pred_idx = 0

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        # 跳过不参与 loss 的位置（特殊 token、非首子词、PAD）
                        # 但这些位置在 pred_ids 中可能存在（如 [CLS]/[SEP]），
                        # 所以 pred_idx 仍需递增
                        if j < len(pred_ids):
                            pred_idx += 1
                        continue
                    # 真实标签
                    gold_seq.append(id2label[gold_id])
                    # 对应位置的预测标签
                    if pred_idx < len(pred_ids):
                        pred_seq.append(id2label.get(pred_ids[pred_idx], "O"))
                    else:
                        pred_seq.append("O")
                    pred_idx += 1

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        # 将 batch 数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播，支持 model(**batch) 解包调用
        _, loss = model(**batch)

        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 处理最后不足 grad_accum 的批次
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系（传入 data_dir，确保与数据集一致）
    labels, label2id, id2label = build_label_schema(data_dir=args.data_dir)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # DataLoader（使用 data_dir 参数，默认指向 peoples_daily 数据集）
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=args.data_dir,
    )

    # 模型（传入 label2id/id2label，与 dataset.py 的标签体系保持一致）
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=args.dropout,
    ).to(device)

    # 分层学习率：BERT 层用基础 lr，分类头用 head_lr_mult 倍
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate.py {'--use_crf' if args.use_crf else ''}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="数据目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
