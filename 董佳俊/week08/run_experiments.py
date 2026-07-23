"""
作业：在 LCQMC 和 BQ Corpus 两个数据集上试验不同文本匹配方法

实验矩阵（2 数据集 x 3 方法 = 6 组）：
  - BiEncoder + CosineEmbeddingLoss
  - BiEncoder + TripletLoss
  - CrossEncoder + CrossEntropyLoss

每组实验：
  1. 从训练集平衡采样 20,000 条（pos:neg = 1:1）
  2. 训练 1 epoch（4 层 BERT，batch_size=32）
  3. 在完整 validation 集上评估
  4. 保存 checkpoint + 训练日志

使用方式：
  cd homework
  python run_experiments.py          # 运行全部 6 组
  python run_experiments.py --dry_run # 仅打印计划，不执行训练
"""

import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows GBK 控制台 UTF-8 兼容处理
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 路径设置 ────────────────────────────────────────────────────────────────
HOMEWORK_DIR = Path(__file__).parent
PROJECT_ROOT = HOMEWORK_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from dataset import (
    PairDataset, TripletDataset, CrossEncoderDataset,
    load_jsonl, encode_single,
)
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

BERT_PATH  = Path("D:/八斗学习内容/pretrain_models/bert-base-chinese")
DATA_DIR   = PROJECT_ROOT / "data"
RESULTS_DIR = HOMEWORK_DIR / "results"
FIG_DIR    = HOMEWORK_DIR / "figures"

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 实验配置 ────────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "dataset": "lcqmc",
        "method":  "biencoder_cosine",
        "label":  "LCQMC + BiEncoder (CosineLoss)",
    },
    {
        "dataset": "lcqmc",
        "method":  "biencoder_triplet",
        "label":  "LCQMC + BiEncoder (TripletLoss)",
    },
    {
        "dataset": "lcqmc",
        "method":  "crossencoder",
        "label":  "LCQMC + CrossEncoder",
    },
    {
        "dataset": "bq_corpus",
        "method":  "biencoder_cosine",
        "label":  "BQ Corpus + BiEncoder (CosineLoss)",
    },
    {
        "dataset": "bq_corpus",
        "method":  "biencoder_triplet",
        "label":  "BQ Corpus + BiEncoder (TripletLoss)",
    },
    {
        "dataset": "bq_corpus",
        "method":  "crossencoder",
        "label":  "BQ Corpus + CrossEncoder",
    },
]

# 默认超参（与 AFQMC 实验保持一致，便于横向对比）
DEFAULT_ARGS = {
    "num_hidden_layers": 4,
    "epochs":            1,
    "batch_size":        32,
    "max_length_bi":     64,
    "max_length_cross":  128,
    "lr":                2e-5,
    "head_lr_mult":      5.0,
    "warmup_ratio":      0.1,
    "grad_accum":        1,
    "margin":            0.3,
    "pool":              "mean",
    "num_train":         20000,   # 平衡采样总数
}


# ── 平衡采样工具 ────────────────────────────────────────────────────────────

def sample_balanced(data_path, n_total, seed=42):
    """
    从 JSONL 数据集中做正负平衡采样。

    返回：采样后的 rows 列表
    """
    rng = random.Random(seed)
    rows = load_jsonl(data_path)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]

    n_each = min(n_total // 2, len(pos), len(neg))
    sampled = rng.sample(pos, n_each) + rng.sample(neg, n_each)
    rng.shuffle(sampled)

    print(f"  平衡采样: 正 {n_each:,} + 负 {n_each:,} = {len(sampled):,} 条 "
          f"（原始: pos={len(pos):,}, neg={len(neg):,}）")
    return sampled


def save_sampled_jsonl(rows, save_path):
    """将采样后的数据保存为临时 JSONL 文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return save_path


# ── BiEncoder CosineLoss 训练 ───────────────────────────────────────────────

def train_biencoder_cosine(config, args, device):
    """BiEncoder + CosineEmbeddingLoss 训练 + 评估"""
    dataset   = config["dataset"]
    out_dir   = RESULTS_DIR / dataset / "biencoder_cosine"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 采样 + 构建 Dataset
    sampled = sample_balanced(DATA_DIR / dataset / "train.jsonl", args["num_train"])
    tmp_train = out_dir / "train_sampled.jsonl"
    save_sampled_jsonl(sampled, tmp_train)

    train_ds = PairDataset(tmp_train, tokenizer, max_length=args["max_length_bi"])
    val_ds   = PairDataset(DATA_DIR / dataset / "validation.jsonl", tokenizer, args["max_length_bi"])

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args["batch_size"], shuffle=False, num_workers=0)

    print(f"  train: {len(train_ds):,} 条  val: {len(val_ds):,} 条")

    # 模型
    model = build_biencoder(str(BERT_PATH), pool=args["pool"],
                            num_hidden_layers=args["num_hidden_layers"]).to(device)

    # 优化器
    bert_params = list(model.bert.parameters())
    head_params = list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args["lr"]},
        {"params": head_params, "lr": args["lr"] * args["head_lr_mult"]},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args["epochs"] // args["grad_accum"]
    warmup_steps = int(total_steps * args["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    # 训练循环
    model.train()
    t0 = time.time()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"[{config['label']}]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        labels = batch["label"].to(device)

        emb_a, emb_b = model(batch_a, batch_b)
        cos_target = (labels.float() * 2 - 1)
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=args["margin"])

        (loss / args["grad_accum"]).backward()
        if (step + 1) % args["grad_accum"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    train_loss = total_loss / total_samples
    train_time = time.time() - t0

    # 评估
    val_metrics = eval_biencoder(model, val_loader, device)
    val_acc  = val_metrics["accuracy"]
    val_f1   = val_metrics["f1"]
    val_thr  = val_metrics["threshold"]

    print(f"  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
          f"val_f1={val_f1:.4f}  threshold={val_thr:.2f}  time={train_time:.0f}s")

    # 保存
    ckpt_path = out_dir / "best_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "threshold":  val_thr,
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "args":       args,
    }, ckpt_path)

    log = {
        "dataset":   dataset,
        "method":    "biencoder_cosine",
        "train_loss": train_loss,
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "threshold":  val_thr,
        "train_time_s": train_time,
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "config": args,
    }
    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return log


# ── BiEncoder TripletLoss 训练 ──────────────────────────────────────────────

def train_biencoder_triplet(config, args, device):
    """BiEncoder + TripletLoss 训练 + 评估"""
    dataset   = config["dataset"]
    out_dir   = RESULTS_DIR / dataset / "biencoder_triplet"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 采样 + 构建 TripletDataset
    sampled = sample_balanced(DATA_DIR / dataset / "train.jsonl", args["num_train"])
    tmp_train = out_dir / "train_sampled.jsonl"
    save_sampled_jsonl(sampled, tmp_train)

    train_ds = TripletDataset(tmp_train, tokenizer, max_length=args["max_length_bi"])
    val_ds   = PairDataset(DATA_DIR / dataset / "validation.jsonl", tokenizer, args["max_length_bi"])

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args["batch_size"], shuffle=False, num_workers=0)

    print(f"  triplet: {len(train_ds):,} 个三元组  val: {len(val_ds):,} 条")

    # 模型
    model = build_biencoder(str(BERT_PATH), pool=args["pool"],
                            num_hidden_layers=args["num_hidden_layers"]).to(device)

    # 优化器
    bert_params = list(model.bert.parameters())
    head_params = list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args["lr"]},
        {"params": head_params, "lr": args["lr"] * args["head_lr_mult"]},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args["epochs"] // args["grad_accum"]
    warmup_steps = int(total_steps * args["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    # 训练循环
    model.train()
    t0 = time.time()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"[{config['label']}]", leave=False)
    for step, batch in enumerate(pbar):
        enc_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        enc_p = {
            "input_ids":      batch["input_ids_p"].to(device),
            "attention_mask": batch["attention_mask_p"].to(device),
            "token_type_ids": batch["token_type_ids_p"].to(device),
        }
        enc_n = {
            "input_ids":      batch["input_ids_n"].to(device),
            "attention_mask": batch["attention_mask_n"].to(device),
            "token_type_ids": batch["token_type_ids_n"].to(device),
        }

        emb_a = model.encode(**enc_a)
        emb_p = model.encode(**enc_p)
        emb_n = model.encode(**enc_n)

        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=args["margin"])

        (loss / args["grad_accum"]).backward()
        if (step + 1) % args["grad_accum"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bs = emb_a.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    train_loss = total_loss / total_samples
    train_time = time.time() - t0

    # 评估（BiEncoder val 用 PairDataset + 阈值搜索）
    val_metrics = eval_biencoder(model, val_loader, device)
    val_acc  = val_metrics["accuracy"]
    val_f1   = val_metrics["f1"]
    val_thr  = val_metrics["threshold"]

    print(f"  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
          f"val_f1={val_f1:.4f}  threshold={val_thr:.2f}  time={train_time:.0f}s")

    # 保存
    ckpt_path = out_dir / "best_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "threshold":  val_thr,
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "args":       args,
    }, ckpt_path)

    log = {
        "dataset":   dataset,
        "method":    "biencoder_triplet",
        "train_loss": train_loss,
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "threshold":  val_thr,
        "train_time_s": train_time,
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "config": args,
    }
    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return log


# ── CrossEncoder 训练 ───────────────────────────────────────────────────────

def train_crossencoder(config, args, device):
    """CrossEncoder + CrossEntropyLoss 训练 + 评估"""
    dataset   = config["dataset"]
    out_dir   = RESULTS_DIR / dataset / "crossencoder"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 采样 + 构建 CrossEncoderDataset
    sampled = sample_balanced(DATA_DIR / dataset / "train.jsonl", args["num_train"])
    tmp_train = out_dir / "train_sampled.jsonl"
    save_sampled_jsonl(sampled, tmp_train)

    train_ds = CrossEncoderDataset(tmp_train, tokenizer, max_length=args["max_length_cross"])
    val_ds   = CrossEncoderDataset(DATA_DIR / dataset / "validation.jsonl", tokenizer, args["max_length_cross"])

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args["batch_size"], shuffle=False, num_workers=0)

    print(f"  train: {len(train_ds):,} 条  val: {len(val_ds):,} 条")

    # 模型
    model = build_crossencoder(str(BERT_PATH),
                               num_hidden_layers=args["num_hidden_layers"]).to(device)

    # 优化器
    bert_params = list(model.bert.parameters())
    head_params = (list(model.dropout.parameters()) + list(model.classifier.parameters()))
    optimizer = AdamW([
        {"params": bert_params, "lr": args["lr"]},
        {"params": head_params, "lr": args["lr"] * args["head_lr_mult"]},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args["epochs"] // args["grad_accum"]
    warmup_steps = int(total_steps * args["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    t0 = time.time()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"[{config['label']}]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        (loss / args["grad_accum"]).backward()
        if (step + 1) % args["grad_accum"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}",
                         acc=f"{total_correct / total_samples:.4f}")

    train_loss = total_loss / total_samples
    train_acc  = total_correct / total_samples
    train_time = time.time() - t0

    # 评估
    val_metrics = eval_crossencoder(model, val_loader, device)
    val_acc = val_metrics["accuracy"]
    val_f1  = val_metrics["f1"]

    print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  time={train_time:.0f}s")

    # 保存
    ckpt_path = out_dir / "best_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "args":       args,
    }, ckpt_path)

    log = {
        "dataset":    dataset,
        "method":     "crossencoder",
        "train_loss": train_loss,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "val_f1":     val_f1,
        "train_time_s": train_time,
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "config": args,
    }
    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return log


# ── 对比汇总 ────────────────────────────────────────────────────────────────

def generate_summary(all_logs):
    """生成所有实验的对比汇总"""
    print(f"\n{'='*80}")
    print(f"{'实验汇总：LCQMC & BQ Corpus 文本匹配方法对比'}")
    print(f"{'='*80}")

    # 表格
    header = f"{'数据集':<14} {'方法':<28} {'Val Acc':>9} {'Val F1':>9} {'阈值':>8} {'耗时':>8}"
    print(f"\n{header}")
    print(f"{'-'*80}")

    for log in all_logs:
        ds   = log["dataset"]
        method = log["method"]
        acc  = log["val_acc"]
        f1   = log["val_f1"]
        thr  = log.get("threshold", None)
        time_min = log["train_time_s"] / 60

        thr_str = f"{thr:.2f}" if thr is not None else "argmax"
        print(f"  {ds:<14} {method:<28} {acc:>9.4f} {f1:>9.4f} {thr_str:>8} {time_min:>7.1f}m")

    # 按数据集分组分析
    print(f"\n{'─'*80}")
    for ds_name in ["lcqmc", "bq_corpus"]:
        ds_logs = [l for l in all_logs if l["dataset"] == ds_name]
        if not ds_logs:
            continue
        best = max(ds_logs, key=lambda x: x["val_f1"])
        print(f"\n  {ds_name.upper()} 最佳方法: {best['method']}  F1={best['val_f1']:.4f}  Acc={best['val_acc']:.4f}")

        # 三种方法排名
        ranked = sorted(ds_logs, key=lambda x: x["val_f1"], reverse=True)
        for i, r in enumerate(ranked):
            print(f"    {i+1}. {r['method']:<28} F1={r['val_f1']:.4f}")

    # 跨数据集分析
    print(f"\n{'─'*80}")
    print("  跨数据集观察：")
    methods = ["biencoder_cosine", "biencoder_triplet", "crossencoder"]
    for m in methods:
        vals = [l["val_f1"] for l in all_logs if l["method"] == m]
        if len(vals) == 2:
            print(f"    {m}: LCQMC={vals[0]:.4f}  BQ={vals[1]:.4f}  "
                  f"平均={sum(vals)/len(vals):.4f}")

    # 保存汇总 JSON
    summary_path = HOMEWORK_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总已保存 → {summary_path}")

    return summary_path


# ── 主入口 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LCQMC & BQ Corpus 文本匹配实验")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印实验计划，不执行训练")
    parser.add_argument("--methods", nargs="+",
                        choices=["biencoder_cosine", "biencoder_triplet", "crossencoder"],
                        default=["biencoder_cosine", "biencoder_triplet", "crossencoder"],
                        help="指定运行的方法（默认全部）")
    parser.add_argument("--datasets", nargs="+",
                        choices=["lcqmc", "bq_corpus"],
                        default=["lcqmc", "bq_corpus"],
                        help="指定数据集（默认全部）")
    parser.add_argument("--num_train", type=int, default=20000,
                        help="平衡采样总数（默认 20000）")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练 epoch 数（默认 1）")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"BERT 路径: {BERT_PATH}")
    print(f"训练样本: {args.num_train}（平衡采样）")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # 筛选实验
    exp_configs = [
        c for c in CONFIGS
        if c["dataset"] in args.datasets and c["method"] in args.methods
    ]

    print(f"\n计划运行 {len(exp_configs)} 组实验：")
    for c in exp_configs:
        print(f"  - {c['label']}")
    print()

    if args.dry_run:
        print("[dry_run] 跳过实际训练。")
        return

    # 更新配置
    run_args = dict(DEFAULT_ARGS)
    run_args.update({
        "num_train": args.num_train,
        "epochs":    args.epochs,
        "batch_size": args.batch_size,
    })

    # ── 逐组运行 ────────────────────────────────────────────────────────────
    all_logs = []
    total_start = time.time()

    for i, config in enumerate(exp_configs):
        print(f"\n{'#'*70}")
        print(f"  [{i+1}/{len(exp_configs)}] {config['label']}")
        print(f"{'#'*70}")

        t0 = time.time()
        try:
            if config["method"] == "biencoder_cosine":
                log = train_biencoder_cosine(config, run_args, device)
            elif config["method"] == "biencoder_triplet":
                log = train_biencoder_triplet(config, run_args, device)
            elif config["method"] == "crossencoder":
                log = train_crossencoder(config, run_args, device)
            else:
                raise ValueError(f"未知方法: {config['method']}")

            all_logs.append(log)
            elapsed = time.time() - t0
            print(f"  [OK] 完成  ({elapsed/60:.1f} min)")

        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 汇总 ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  全部实验完成！总耗时: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")

    if all_logs:
        generate_summary(all_logs)
    else:
        print("  没有成功的实验记录。")


if __name__ == "__main__":
    main()
