"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2.5-0.5B-Instruct 做 NER

教学重点：
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表
     与分类任务的区别：TARGET 是多 token 的结构化 JSON，而非单个类别名
  2. Loss masking：同样只在 JSON 输出部分计算 loss，prompt 全为 -100
  3. LoRA 高效微调：参数量约 0.22%，与全量微调的对比（--full_ft 开关）
  4. 生成式 NER vs 序列标注（BERT+CRF）：各自的优劣和适用场景

使用方式：
  python train_sft.py                        # LoRA，全量训练数据（默认）
  python train_sft.py --num_train 2000       # LoRA，2000 条快速演示
  python train_sft.py --epochs 1             # 快速验证流程

  # 全量微调（需显存 ≥ 16GB）
  python train_sft.py --full_ft --lr 2e-5

依赖：
  pip install torch transformers peft tqdm   # LoRA 模式
  pip install torch transformers tqdm        # 全量微调模式（不需要 peft）
"""

import os
import argparse
import json
import random
import time
import warnings
from pathlib import Path

import torch
# bitsandbytes 内部使用了已弃用的 torch._check_is_size，抑制 FutureWarning
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
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

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "peoples_daily"
_MODEL_05B = Path(r"D:\mydocs\pretrained_models\Qwen2.5-0.5B-Instruct")
_MODEL_7B  = Path(r"D:\mydocs\pretrained_models\Qwen2.5-7B-Instruct")
_MODEL_LEGACY = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"  # 旧路径回退
MODEL_PATH = _MODEL_05B if _MODEL_05B.exists() else _MODEL_LEGACY
OUTPUT_DIR = ROOT / "outputs"

ENTITY_TYPES = [
    "PER",   # 人名
    "ORG",   # 组织机构
    "LOC",   # 地点
]

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地点）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def bio_to_entities(tokens: list, ner_tags: list) -> list[dict]:
    """将 BIO 标签序列解析为实体列表。

    输入：tokens=["海","钓",...], ner_tags=["O","B-LOC","I-LOC",...]
    输出：[{"text": "厦门", "type": "LOC", "start": 0, "end": 1}, ...]
    """
    entities = []
    current_type = None
    current_start = 0
    current_chars = []

    for i, tag in enumerate(ner_tags):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append({
                    "text": "".join(current_chars),
                    "type": current_type,
                    "start": current_start,
                    "end": i - 1,
                })
            current_type = tag[2:]
            current_start = i
            current_chars = [tokens[i]]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_chars.append(tokens[i])
        else:
            if current_type is not None:
                entities.append({
                    "text": "".join(current_chars),
                    "type": current_type,
                    "start": current_start,
                    "end": i - 1,
                })
                current_type = None
                current_chars = []

    if current_type is not None:
        entities.append({
            "text": "".join(current_chars),
            "type": current_type,
            "start": current_start,
            "end": len(ner_tags) - 1,
        })

    return entities


def record_to_target(record: dict) -> str:
    """把 peoples_daily BIO 格式转为 SFT 目标 JSON 字符串。

    输入：{"tokens": [...], "ner_tags": ["B-PER", "I-PER", "O", ...]}
    输出：'{"entities": [{"text": "张三", "type": "PER"}, ...]}'
    """
    entities = bio_to_entities(record["tokens"], record["ner_tags"])
    return json.dumps(
        {"entities": [{"text": e["text"], "type": e["type"]} for e in entities]},
        ensure_ascii=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    把 NER 数据转换为 chat-format SFT 训练样本（兼容 cluener span 和 peoples_daily BIO 格式）。

    与分类任务的关键区别：
      - 分类：TARGET = "科技"（1~2 个 token，极短）
      - NER：TARGET = '{"entities": [...]}' （20~150 个 token，结构化 JSON）

    Loss mask 结构：
      ┌──────────────────────────────────────────────────────────────┐
      │ <system>...<user>{text}<assistant>\n                         │  → -100
      │ {"entities": [{"text": "浙商银行", "type": "company"}]} <EOS>│  → 真实 id
      └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target = record_to_target(item)

        # ── Step 1：构建 prompt 文本（tokenize=False 兼容 transformers 5.x）──
        # 人民网数据：text = "".join(tokens)；cluener 兼容：直接用 item["text"]
        text = "".join(item["tokens"]) if "tokens" in item else item["text"]
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # ── Step 2：response = JSON 字符串 + EOS ──────────────────────────────
        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接 + 截断 ───────────────────────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # ── Step 4：loss mask：prompt 全 -100，只在 JSON 部分计算 loss ──────
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
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部 20864 条（默认）")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=1,    type=int,
                        help="batch size（8GB 显存建议 1，16GB+ 可用 2-4）")
    parser.add_argument("--grad_accum",  default=16,   type=int,
                        help="梯度累积步数（配合 batch_size=1 保持等效 batch=16）")
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=192,  type=int,
                        help="序列最大长度（8GB 显存建议 128-192，人民网 P95=97 字 + JSON 输出）")
    # 全量微调开关
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有参数（需显存 ≥ 16GB）")
    # QLoRA 4-bit 量化（适配 7B 模型到 8GB 显存）
    parser.add_argument("--qlora",       action="store_true",
                        help="QLoRA 模式：4-bit NF4 量化 + LoRA（用于 7B 模型，需 bitsandbytes）")
    parser.add_argument("--model_size",  choices=["0.5B", "7B"], default="0.5B",
                        help="选择模型规格（0.5B 或 7B），自动切换模型路径和量化策略")
    parser.add_argument("--qlora_double_quant", action="store_true", default=True,
                        help="QLoRA 双重量化（进一步节省显存，默认开启）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r",      default=8,    type=int)
    parser.add_argument("--lora_alpha",  default=16,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 模型路径选择 ──────────────────────────────────────────────────────────
    if args.model_size == "7B":
        if not _MODEL_7B.exists():
            raise FileNotFoundError(f"7B 模型不存在: {_MODEL_7B}")
        model_path = str(_MODEL_7B)
        args.qlora = True  # 7B 必须用 QLoRA
        if args.max_length == 192:  # 未手动指定 → 提高到 256（NER JSON target 长）
            args.max_length = 256
        if args.batch_size == 1 and args.grad_accum == 16:  # 默认参数 → 更保守
            args.grad_accum = 8   # 与 week06 一致
        ckpt_dir = output_dir / "sft_adapter_7b"
    else:
        model_path = str(_MODEL_05B) if _MODEL_05B.exists() else str(_MODEL_LEGACY)
        ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "全量微调" if args.full_ft else ("QLoRA 4-bit" if args.qlora else "LoRA 微调")
    print(f"使用设备: {device}  |  微调模式: {mode_str}  |  模型: {args.model_size}")
    print(f"模型路径: {model_path}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    print(f"原始训练集: {len(train_raw)} 条 | 验证集: {len(val_raw)} 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 过滤超长样本 ──────────────────────────────────────────────────────────
    def _token_len(record):
        text = "".join(record["tokens"]) if "tokens" in record else record["text"]
        target = record_to_target(record)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        resp_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        return len(prompt_ids) + len(resp_ids)

    before_train = len(train_raw)
    train_raw = [r for r in train_raw if _token_len(r) <= args.max_length]
    val_raw = [r for r in val_raw if _token_len(r) <= args.max_length]
    if before_train != len(train_raw):
        print(f"过滤超长样本: {before_train} -> {len(train_raw)} ({before_train - len(train_raw)} 条被移除）")
    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:300], tokenizer, args.max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    use_fp16 = device.type == "cuda" and not args.qlora  # QLoRA 自带量化，不额外设 dtype
    model_kwargs = {"trust_remote_code": True}

    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,       # 与 week06 成功配置一致
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"        # 关键：让 transformers 自动分配层到 GPU/CPU
        print(f"加载 base model: {model_path}  (QLoRA 4-bit NF4, device_map=auto)")
    else:
        model_kwargs["torch_dtype"] = model_dtype
        print(f"加载 base model: {model_path}  (dtype={model_dtype})")

    if torch.cuda.is_available():
        free_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU 总显存: {free_mb:.0f} MB")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],  # 与 week06 一致，只用 2 个模块
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False  # 训练时关闭 KV cache，节省显存
        model.print_trainable_parameters()

    # device_map="auto" 时模型已自动放置，不需要手动 to(device)
    if not args.qlora:
        model = model.to(device)

    # ── 显存监控 ────────────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU 显存: 已分配={allocated:.0f}MB, 已保留={reserved:.0f}MB")
        if reserved > 7500:
            print("  [!] 警告：显存使用超过 7.5GB，训练中可能 OOM，请减小 batch_size 或 max_length")

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
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
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  [OK] 最优{ckpt_label}已保存 -> {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag  = "full_ft" if args.full_ft else ("sft_7b" if args.model_size == "7B" else "sft")
    log_path = output_dir / "logs" / f"train_{log_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看 entity F1 与多方对比")


if __name__ == "__main__":
    main()
