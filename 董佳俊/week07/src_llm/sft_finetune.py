"""
LLM 指令微调做文本分类 —— Qwen2.5 + LoRA

将分类任务改写成生成式格式：
  输入: "请判断以下文本的情感倾向：{text}"
  输出: "正面" 或 "负面"

训练方式（二选一）：
  --lora   : LoRA 高效微调（默认，显存 ~4GB）
  --full   : 全量微调（显存需求 ≥ 16GB）

将文本分类任务转化为生成式格式：输入为文本+指令，输出为目标类别标签。Loss 只计算在回答部分，prompt 部分用 -100 mask 掉。

用法：
  python src_llm/sft_finetune.py                         # LoRA，全量数据
  python src_llm/sft_finetune.py --num_train 200         # 200 条快速演示
  python src_llm/sft_finetune.py --full --epochs 1       # 全量微调快速验证
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
CKPT_DIR = OUTPUT_DIR / "sft_ckpt"

LABEL_WORDS = {0: "负面", 1: "正面"}
LABEL_IDS = {"负面": 0, "正面": 1}

INSTRUCTION = '请判断以下文本的情感倾向，只回答「正面」或「负面」，不要解释。\n文本：'


# ── 数据转换：分类样本 → 生成式格式 ──

def format_sample(text: str, label: int | None = None) -> dict:
    """将一条文本分类样本转为 chat 格式。

    训练时: prompt = instruction + text, completion = label_word
    推理时: 只有 prompt，无 completion
    """
    prompt = INSTRUCTION + text
    label_word = LABEL_WORDS[label] if label is not None else None
    return {"prompt": prompt, "completion": label_word}


# ── Dataset ──

class SFTDataset(Dataset):
    """生成式分类 Dataset。

    核心技巧：
      - 将 prompt + completion 拼接在一起 tokenize
      - prompt 部分的 label 设为 -100（不计算 loss）
      - completion 部分的 label 保留 token id
    """

    def __init__(self, records: list[dict], tokenizer, max_len: int = 512):
        self.samples = []
        for r in records:
            fmt = format_sample(r["text"], r["label"])
            full_text = fmt["prompt"] + "\n回答：" + fmt["completion"]

            # tokenize 完整文本
            enc = tokenizer(full_text, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # tokenize prompt 部分，确定 mask 边界
            prompt_enc = tokenizer(fmt["prompt"] + "\n回答：", max_length=max_len,
                                   truncation=True, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            # label: prompt 部分 = -100，completion 部分 = token id
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            # 忽略 pad token
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


# ── 推理 & 评估 ──

@torch.no_grad()
def sft_evaluate(model, tokenizer, records: list[dict], device, max_new: int = 16) -> dict:
    """对生成式模型做分类评估：生成文本 → 解析标签 → 计算 metrics。"""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    y_true, y_pred = [], []

    for r in records:
        fmt = format_sample(r["text"])
        prompt = fmt["prompt"] + "\n回答："
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # 简单规则匹配（取第一个出现的"正面"或"负面"）
        if "正面" in generated:
            pred = 1
        elif "负面" in generated:
            pred = 0
        else:
            pred = -1  # 无法解析

        y_true.append(r["label"])
        y_pred.append(pred)

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    if not valid:
        return {"accuracy": 0.0, "f1": 0.0, "parse_rate": 0.0}

    yt, yp = zip(*valid)
    return {
        "accuracy": round(accuracy_score(yt, yp), 6),
        "f1": round(f1_score(yt, yp, average="macro", zero_division=0), 6),
        "parse_rate": round(len(valid) / len(records), 6),
    }


# ── 训练一个 epoch ──

def sft_train_epoch(model, loader, optimizer, scheduler, device, grad_accum, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
    from data_loader import load_dataset
    train_records = load_dataset("train")
    val_records = load_dataset("validation")
    test_records = load_dataset("test")

    if args.num_train and args.num_train < len(train_records):
        import random
        random.seed(42)
        train_records = random.sample(train_records, args.num_train)

    print(f"训练数据: {len(train_records)} 条, 验证: {len(val_records)} 条")

    # —— 模型 & Tokenizer ——
    model_path = str(args.model_path) if args.model_path else "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # LoRA vs 全量
    if args.lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"], lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_cfg)
            print(f"  LoRA 已注入 (r={args.lora_r}, alpha={args.lora_alpha})")
        except ImportError:
            print("  ⚠ peft 未安装，回退到全量微调")
            args.lora = False

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数总量: {total / 1e6:.1f}M, 可训练: {trainable / 1e6:.1f}M ({trainable / max(total, 1) * 100:.1f}%)")

    # —— DataLoader ——
    def _make_loader(records, shuffle=False):
        ds = SFTDataset(records, tokenizer, max_len=args.max_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(train_records, shuffle=True)
    val_loader = _make_loader(val_records)

    # —— 优化器 ——
    optimizer = AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.01}],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    print(f"  总步数: {total_steps}")

    # —— 训练 ——
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    tag = "lora" if args.lora else "full"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = sft_train_epoch(model, train_loader, optimizer, scheduler, device,
                                     args.grad_accum, epoch, args.epochs)
        val_metrics = sft_evaluate(model, tokenizer, val_records[:100], device)  # 采样评估加速
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f} | {elapsed:.0f}s")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt_path = CKPT_DIR / f"sft_cls_{tag}"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"    ★ 最优 checkpoint → {ckpt_path}")

    # —— 测试集评估 ——
    print(f"\n{'=' * 60}")
    print(f"  测试集评估 (SFT-{tag})")
    test_metrics = sft_evaluate(model, tokenizer, test_records, device)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1:       {test_metrics['f1']:.4f}")
    print(f"  解析率:   {test_metrics['parse_rate']:.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": f"Qwen2.5-SFT-{tag}",
        "test_metrics": test_metrics,
        "best_val_f1": best_f1,
    }
    out_path = LOG_DIR / f"sft_{tag}_cls.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存 → {out_path}")
    print(f"\n  下一步：python compare_all.py")


def parse_args():
    p = argparse.ArgumentParser(description="LLM SFT 文本分类")
    p.add_argument("--lora", action="store_true", default=True, help="LoRA 微调（默认开启）")
    p.add_argument("--full", action="store_true", help="全量微调（关闭 LoRA）")
    p.add_argument("--model_path", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_train", type=int, default=None, help="限制训练样本数（用于快速演示）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full:
        args.lora = False
    main()
