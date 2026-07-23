"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2.5-0.5B-Instruct 做 NER

教学重点：
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表
     与分类任务的区别：TARGET 是多 token 的结构化 JSON，而非单个类别名
  2. Loss masking：同样只在 JSON 输出部分计算 loss，prompt 全为 -100
  3. LoRA 高效微调：参数量约 0.22%，与全量微调的对比（--full_ft 开关）
  4. 生成式 NER vs 序列标注（BERT+CRF）：各自的优劣和适用场景
  5. --dataset 参数：支持 cluener（10类）和 peoples_daily（3类：PER/ORG/LOC）

使用方式：
  python train_sft.py --dataset peoples_daily                        # LoRA 微调（默认参数）
  python train_sft.py --dataset peoples_daily --num_train 5000       # 5000 条训练
  python train_sft.py --dataset peoples_daily --epochs 2 --batch_size 2 --grad_accum 4 --lora_r 4

  # 全量微调（需显存 ≥ 16GB）
  python train_sft.py --dataset peoples_daily --full_ft --lr 2e-5

依赖：
  pip install torch transformers peft tqdm
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT = Path(__file__).parent.parent
PRETRAIN_DIR = Path("D:/BaiduNetdiskDownload/八斗学院ai大模型/AI大模型培训部分/pretrain_models")
ORIGINAL_DATA = Path(
    "D:/aipy/AI大模型培训部分/week7序列标注问题_0530/"
    "week7 序列标注问题/序列标注项目/data"
)
MODEL_PATH = PRETRAIN_DIR / "Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

# ══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════════════

ENTITY_TYPES_CLUENER = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
]

SYSTEM_PROMPT_CLUENER = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：address（地址）、book（书名）、company（公司）、"
    "game（游戏）、government（政府机构）、movie（影视作品）、name（人名）、"
    "organization（组织机构）、position（职位）、scene（景点/场所）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)

ENTITY_TYPES_PD = ["PER", "ORG", "LOC"]

SYSTEM_PROMPT_PD = (
    "你是一个命名实体识别助手。从文本中识别人名、组织机构和地名，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


# ══════════════════════════════════════════════════════════════════════════════
# 数据转换
# ══════════════════════════════════════════════════════════════════════════════

def record_to_target_cluener(record: dict) -> str:
    """cluener span 格式 → SFT 目标 JSON 字符串。"""
    entities = []
    for etype, surfaces in (record.get("label") or {}).items():
        for surface in surfaces:
            entities.append({"text": surface, "type": etype})
    return json.dumps({"entities": entities}, ensure_ascii=False)


def record_to_target_pd(record: dict) -> str:
    """peoples_daily BIO 格式 → SFT 目标 JSON 字符串。"""
    # 延迟导入，避免循环依赖
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from dataset import bio_tags_to_entities

    entities = bio_tags_to_entities(record["tokens"], record["ner_tags"])
    return json.dumps({"entities": entities}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """把 NER 数据转换为 chat-format SFT 训练样本。

    Loss mask 结构：
      ┌──────────────────────────────────────────────────────────────┐
      │ <system>...<user>{text}<assistant>\n                         │  → -100
      │ {"entities": [{"text": "...", "type": "..."}]} <EOS>        │  → 真实 id
      └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, data, tokenizer, max_length, system_prompt, record_to_target):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self._record_to_target = record_to_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text") or "".join(item["tokens"])
        target = self._record_to_target(item)

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）")
    parser.add_argument("--dataset", type=str, choices=["cluener", "peoples_daily"],
                        default="peoples_daily", help="数据集选择")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train", default=5000, type=int,
                        help="训练样本数，-1 使用全部")
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length", default=192, type=int,
                        help="序列最大长度")
    parser.add_argument("--full_ft", action="store_true",
                        help="全量微调：跳过 LoRA，更新所有参数（需显存 ≥ 16GB）")
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    ds = args.dataset

    # ── 数据集配置 ────────────────────────────────────────────────────────────
    if ds == "peoples_daily":
        data_dir = ORIGINAL_DATA / "peoples_daily"
        system_prompt = SYSTEM_PROMPT_PD
        record_to_target = record_to_target_pd
    else:
        data_dir = ORIGINAL_DATA / "cluener"
        system_prompt = SYSTEM_PROMPT_CLUENER
        record_to_target = record_to_target_cluener

    output_dir = Path(args.output_dir)
    ckpt_tag = "sft_full_ckpt" if args.full_ft else "sft_adapter"
    ckpt_dir = output_dir / f"{ckpt_tag}_{ds}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  数据集: {ds}  |  微调模式: {mode_str}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集（前300条）: 300 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length,
                               system_prompt, record_to_target)
    val_dataset   = SFTDataset(val_raw[:300], tokenizer, args.max_length,
                               system_prompt, record_to_target)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || "
              f"trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]",
                    leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens      = (labels != -100).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens   = (labels != -100).sum().item()
                val_loss   += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  "
                  f"(val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag  = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"train_{log_tag}_{ds}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py --dataset {ds}")


if __name__ == "__main__":
    main()
