"""
四路文本分类方案对比评估

对比四种方法在相同验证集上的效果：
  1. BERT fine-tune（判别式，监督微调）
  2. LLM zero-shot（生成式，无需训练）
  3. LLM few-shot（生成式，每类 2 个示例）
  4. LLM SFT LoRA（生成式，LoRA 高效微调）
  5. LLM SFT 全量微调（生成式，监督微调）

使用方式：
  python compare_all.py                          # 评估全部方法
  python compare_all.py --num_samples 500        # 采样 500 条
  python compare_all.py --bert_ckpt None         # 只跑 LLM 相关
  python compare_all.py --demo                   # 只跑 5 条快速演示
  python compare_all.py --analysis               # 详细分析 zero-shot 各类别表现
  python compare_all.py --few_shot               # 启用 few-shot（每类 2 示例）

依赖：
  pip install torch transformers peft scikit-learn tqdm
"""

import os
import sys
import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# 绘图依赖
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
matplotlib.rcParams["axes.unicode_minus"] = False

def _find_chinese_font():
    for name in ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            return name
    return None

_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
else:
    print("[警告] 未找到中文字体，图表标签可能显示异常")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── 项目路径配置 ───────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
SRC_DIR     = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
DATA_DIR    = ROOT / "data"
BERT_PATH   = ROOT.parent / "pretrain_models" / "bert-base-chinese"
LLM_PATH    = ROOT.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR  = ROOT / "outputs"
FEATURES_ALL_DIR = OUTPUT_DIR / "features_all"
# BERT 三个池化策略的 checkpoint
BERT_CKPT_CLS  = OUTPUT_DIR / "checkpoints" / "best_cls.pt"
BERT_CKPT_MEAN = OUTPUT_DIR / "checkpoints" / "best_mean.pt"
BERT_CKPT_MAX  = OUTPUT_DIR / "checkpoints" / "best_max_weighted.pt"
LORA_CKPT   = OUTPUT_DIR / "sft_adapter"
FULL_CKPT   = OUTPUT_DIR / "sft_full_ckpt"

# ── 类别名 ──────────────────────────────────────────────────────────────────────
LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


# ══════════════════════════════════════════════════════════════════════════════
# 混淆矩阵绘图函数
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(preds, labels, save_path: Path):
    """绘制并保存混淆矩阵热力图（绝对计数 + 按行归一化）。"""
    FEATURES_ALL_DIR.mkdir(parents=True, exist_ok=True)

    label_ids = list(range(len(LABEL_NAMES)))
    cm = confusion_matrix(labels, preds, labels=label_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, annot_kws={"size": 7})
    axes[0].set_title("混淆矩阵（绝对计数）")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, annot_kws={"size": 7},
                vmin=0, vmax=1)
    axes[1].set_title("混淆矩阵（按行归一化，对角线 = Recall）")
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵 → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 数据准备：加载相同的验证集样本（保证对比公平）
# ══════════════════════════════════════════════════════════════════════════════

def load_val_samples(data_dir: Path, num_samples: int, seed: int, full: bool = False):
    """加载验证集并随机采样，保证四种方法评估的是相同样本。

    Args:
        full: 如果为 True，num_samples 被忽略，返回全部验证集（用于生成可靠混淆矩阵）
    """
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)

    random.seed(seed)
    if full:
        samples = val_data  # 不采样，使用全部验证集
    else:
        samples = random.sample(val_data, min(num_samples, len(val_data)))
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    # 构建结构化样本列表
    structured = []
    for item in samples:
        structured.append({
            "text": item["sentence"],
            "true_label": item["label"],
            "true_name": id2name[item["label"]],
        })
    return structured, id2name


# ══════════════════════════════════════════════════════════════════════════════
# 方法1：BERT fine-tune 评估（判别式分类）
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bert(samples, id2name, bert_path, bert_ckpt_path, device, pool="cls"):
    """评估 BERT fine-tune 模型（与 src/evaluate.py 相同的 checkpoint 格式）。"""
    from transformers import BertTokenizer
    from model import build_model

    if bert_ckpt_path is None or not Path(bert_ckpt_path).exists():
        return {"accuracy": None, "correct": 0, "total": 0,
                "elapsed": 0, "status": "checkpoint not found", "per_class": {}}

    ckpt = torch.load(bert_ckpt_path, map_location=device, weights_only=False)
    pool = ckpt.get("pool", pool)

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = build_model(bert_path, num_labels=len(id2name), pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  BERT checkpoint: {bert_ckpt_path}  (pool={pool}, val_acc={ckpt.get('val_acc', 'N/A')})")

    correct, total = 0, 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds, all_labels = [], []
    t0 = time.time()

    for item in tqdm(samples, desc="BERT", leave=False):
        encoding = tokenizer(
            item["text"], max_length=128, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
        pred_id = logits.argmax(dim=-1).item()
        true_label = item["true_label"]
        true_name = item["true_name"]
        per_class[true_name]["total"] += 1
        all_preds.append(pred_id)
        all_labels.append(true_label)
        if pred_id == true_label:
            correct += 1
            per_class[true_name]["correct"] += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "correct": correct, "total": total,
            "elapsed": elapsed, "status": "OK", "per_class": dict(per_class),
            "preds": all_preds, "labels": all_labels}


# ══════════════════════════════════════════════════════════════════════════════
# 方法2：LLM zero-shot 评估（生成式，无需训练）
# ══════════════════════════════════════════════════════════════════════════════

def load_llm_base(model_path: str, device: torch.device):
    """加载原生 LLM（未微调）。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  [zero-shot] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def llm_classify_one(text, model, tokenizer, device, max_new_tokens=8):
    """LLM 单条分类，返回原始输出字符串。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"新闻标题：{text}\n类别："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None, top_p=None, top_k=None,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output):
    """从 LLM 输出中提取类别名（模糊匹配）。"""
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Few-shot 支持：从训练集中每类抽取示例构建 prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_few_shot_prompt(train_path: Path, shots_per_class: int = 2) -> str:
    """从训练集每类抽取 `shots_per_class` 条样本，构建 few-shot 示例文本。"""
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    # 按类别分组
    by_class = defaultdict(list)
    for item in train_data:
        by_class[item["label"]].append(item["sentence"])

    # 每类随机选 `shots_per_class` 条
    examples_lines = []
    for label_id, sentences in by_class.items():
        name = id2name[label_id]
        sampled = random.sample(sentences, min(shots_per_class, len(sentences)))
        for sent in sampled:
            examples_lines.append(f"新闻标题：{sent}\n类别：{name}")

    random.shuffle(examples_lines)
    examples_text = "\n\n".join(examples_lines)

    return (
        "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
        "只输出类别名称，不要输出任何其他内容。\n"
        "可选类别：" + "、".join(LABEL_NAMES) +
        "\n\n以下是一些分类示例：\n\n" + examples_text +
        "\n\n现在请分类以下新闻标题："
    )


def llm_classify_with_prompt(text, model, tokenizer, device, prompt: str, max_new_tokens=8):
    """使用自定义 prompt 对单条文本分类。"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": f"新闻标题：{text}\n类别："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None, top_p=None, top_k=None,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_llm_zero_shot(samples, model_path, device, few_shot_prompt=None):
    """评估 LLM zero-shot/few-shot。few_shot_prompt 为 None 时是 zero-shot。"""
    if not Path(model_path).exists():
        return {"accuracy": None, "correct": 0, "total": 0,
                "elapsed": 0, "unparseable": 0, "status": "model not found",
                "per_class": {}}

    model, tokenizer = load_llm_base(model_path, device)

    # 名称 -> id 映射（用于混淆矩阵）
    name_to_id = {name: i for i, name in enumerate(LABEL_NAMES)}

    correct, total, unparseable = 0, 0, 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds, all_labels = [], []
    t0 = time.time()

    for item in tqdm(samples, desc="Few-shot" if few_shot_prompt else "Zero-shot", leave=False):
        if few_shot_prompt:
            raw_output = llm_classify_with_prompt(item["text"], model, tokenizer, device, few_shot_prompt)
        else:
            raw_output = llm_classify_one(item["text"], model, tokenizer, device)
        pred_name = parse_prediction(raw_output)
        true_label = item["true_label"]
        true_name = item["true_name"]

        per_class[true_name]["total"] += 1
        all_labels.append(true_label)
        all_preds.append(name_to_id.get(pred_name, -1))

        if pred_name == true_name:
            correct += 1
            per_class[true_name]["correct"] += 1
        if pred_name is None:
            unparseable += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "correct": correct, "total": total,
            "unparseable": unparseable, "elapsed": elapsed, "status": "OK",
            "per_class": dict(per_class), "preds": all_preds, "labels": all_labels}


# ══════════════════════════════════════════════════════════════════════════════
# 方法3 & 4：LLM SFT LoRA / 全量微调 评估
# ══════════════════════════════════════════════════════════════════════════════

def load_sft_model(model_path: str, ckpt_dir: str, device: torch.device):
    """加载 SFT 微调后的模型（自动识别 LoRA / 全量微调）。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    # 检查是否有有效模型文件
    has_model = any((ckpt_path / f).exists() for f in ["model.safetensors", "pytorch_model.bin", "model.bin"])
    if not is_lora and not has_model:
        return None, None  # 全量微调 checkpoint 不存在

    if is_lora:
        from peft import PeftModel
        print(f"  [SFT-LoRA] 加载 base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"  [SFT-LoRA] 加载 adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"  [SFT-Full] 加载 checkpoint: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_llm_sft(samples, model_path, ckpt_dir, device, ckpt_type="SFT"):
    """评估 LLM SFT 模型（LoRA 或全量微调）。"""
    if not Path(ckpt_dir).exists():
        return {"accuracy": None, "correct": 0, "total": 0,
                "elapsed": 0, "unparseable": 0, "status": f"{ckpt_type} checkpoint not found",
                "per_class": {}}

    model, tokenizer = load_sft_model(model_path, ckpt_dir, device)
    if model is None or tokenizer is None:
        return {"accuracy": None, "correct": 0, "total": 0,
                "elapsed": 0, "unparseable": 0, "status": f"{ckpt_type} checkpoint invalid",
                "per_class": {}}

    name_to_id = {name: i for i, name in enumerate(LABEL_NAMES)}
    correct, total, unparseable = 0, 0, 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds, all_labels = [], []
    t0 = time.time()

    for item in tqdm(samples, desc=ckpt_type, leave=False):
        raw_output = llm_classify_one(item["text"], model, tokenizer, device)
        pred_name = parse_prediction(raw_output)

        true_label = item["true_label"]
        true_name = item["true_name"]
        per_class[true_name]["total"] += 1
        all_labels.append(true_label)
        all_preds.append(name_to_id.get(pred_name, -1))
        if pred_name == true_name:
            correct += 1
            per_class[true_name]["correct"] += 1
        if pred_name is None:
            unparseable += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "correct": correct, "total": total,
            "unparseable": unparseable, "elapsed": elapsed, "status": "OK",
            "per_class": dict(per_class), "preds": all_preds, "labels": all_labels}


# ══════════════════════════════════════════════════════════════════════════════
# 主函数：运行四种方法并对比
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="四路文本分类方案对比评估")
    # 数据参数
    parser.add_argument("--data_dir",    default=str(DATA_DIR),  type=str)
    parser.add_argument("--num_samples", default=200,  type=int,
                        help="验证集采样数（默认 200）")
    parser.add_argument("--seed",        default=42,   type=int)
    # 模型路径
    parser.add_argument("--bert_path",   default=str(BERT_PATH),  type=str)
    parser.add_argument("--llm_path",    default=str(LLM_PATH),   type=str)
    parser.add_argument("--bert_ckpt",   default=str(BERT_CKPT_CLS), type=str,
                        help="BERT checkpoint 路径，None 则跳过 BERT 评估")
    parser.add_argument("--pool",        default="cls",
                        choices=["cls", "mean", "max"],
                        help="BERT 池化策略（默认 cls）")
    # 演示模式
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条快速演示")
    # 演示模式改为全量验证集（与 evaluate.py 保持一致）
    parser.add_argument("--full",        action="store_true",
                        help="使用完整验证集（10000条）生成混淆矩阵")
    # Few-shot 模式
    parser.add_argument("--few_shot",    action="store_true",
                        help="启用 few-shot 评估（每类 2 个示例）")
    parser.add_argument("--shots",       default=2,  type=int,
                        help="Few-shot 每类示例数（默认 2）")
    # 分析模式
    parser.add_argument("--analysis",    action="store_true",
                        help="输出每个类别的准确率细表")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"随机种子: {args.seed}\n")

    # 加载相同的验证集样本
    n = 5 if args.demo else args.num_samples
    samples, id2name = load_val_samples(Path(args.data_dir), n, args.seed, full=args.full)
    print(f"评估样本数: {len(samples)}\n")

    # Few-shot prompt 预构建（复用同一个模型，只改 prompt）
    few_shot_prompt = None
    if args.few_shot:
        print("构建 few-shot prompt（每类 {} 个示例）...".format(args.shots))
        few_shot_prompt = build_few_shot_prompt(DATA_DIR / "train.json", args.shots)
        print(f"  prompt 长度: {len(few_shot_prompt)} 字符\n")

    # ── 依次评估 ─────────────────────────────────────────────────────────────
    results = {}

    # 方法1：BERT fine-tune（评估三种池化策略）
    print(f"\n{'='*60}")
    print("方法1：BERT fine-tune（判别式分类，cls/mean/max 三种池化）")
    print(f"{'='*60}")

    # 尝试三种池化策略的 checkpoint
    pool_configs = [
        ("BERT cls",  BERT_CKPT_CLS,  "cls"),
        ("BERT mean", BERT_CKPT_MEAN, "mean"),
        ("BERT max",  BERT_CKPT_MAX,  "max"),
    ]
    for name, ckpt_path, pool in pool_configs:
        if ckpt_path.exists():
            results[name] = evaluate_bert(
                samples, id2name, args.bert_path, str(ckpt_path), device, pool,
            )
        else:
            results[name] = {"accuracy": None, "correct": 0, "total": 0,
                              "elapsed": 0, "status": "checkpoint not found", "per_class": {}}

    # 方法2：LLM zero-shot
    print(f"\n{'='*60}")
    print("方法2：LLM zero-shot（生成式，无需训练）")
    print(f"{'='*60}")
    results["LLM zero-shot"] = evaluate_llm_zero_shot(samples, args.llm_path, device, few_shot_prompt=None)

    # 方法3：LLM few-shot（与 zero-shot 共用模型，只是 prompt 不同）
    if args.few_shot:
        print(f"\n{'='*60}")
        print(f"方法3：LLM few-shot（每类 {args.shots} 示例）")
        print(f"{'='*60}")
        results["LLM few-shot"] = evaluate_llm_zero_shot(samples, args.llm_path, device, few_shot_prompt=few_shot_prompt)

    # 方法4：LLM SFT LoRA
    print(f"\n{'='*60}")
    print("方法4：LLM SFT LoRA（生成式，LoRA 高效微调）")
    print(f"{'='*60}")
    results["LLM SFT LoRA"] = evaluate_llm_sft(
        samples, args.llm_path, str(LORA_CKPT), device, "SFT-LoRA",
    )

    # 方法5：LLM SFT 全量微调
    print(f"\n{'='*60}")
    print("方法5：LLM SFT 全量微调（生成式，监督微调）")
    print(f"{'='*60}")
    results["LLM SFT Full"] = evaluate_llm_sft(
        samples, args.llm_path, str(FULL_CKPT), device, "SFT-Full",
    )

    # ── 打印对比表格 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("文本分类多路对比结果")
    print(f"{'='*60}")
    print(f"{'方法':<22} {'准确率':>8} {'正确/总数':>12} {'无法解析':>10} {'耗时':>8}")
    print(f"{'-'*70}")

    for method, r in results.items():
        if r["status"] != "OK":
            acc_str = f"（{r['status']}）"
            correct_str = "-"
            unparseable_str = "-"
            elapsed_str = "-"
        else:
            acc_str = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
            correct_str = f"{r['correct']}/{r['total']}"
            unparseable_str = f"{r.get('unparseable', 0)}"
            elapsed_str = f"{r['elapsed']:.1f}s"
        print(f"{method:<22} {acc_str:>8} {correct_str:>12} {unparseable_str:>10} {elapsed_str:>8}")

    print(f"{'-'*70}")

    # ── 生成混淆矩阵热力图 ───────────────────────────────────────────────────
    FEATURES_ALL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n生成混淆矩阵...")
    for method, r in results.items():
        if r["status"] != "OK" or r.get("accuracy") is None:
            continue
        if "preds" in r and "labels" in r and len(r["preds"]) > 0:
            safe_name = method.replace(" ", "_")
            save_path = FEATURES_ALL_DIR / f"confusion_matrix_{safe_name}.png"
            try:
                plot_confusion_matrix(r["preds"], r["labels"], save_path)
            except Exception as e:
                print(f"  {method} 混淆矩阵生成失败: {e}")

    # ── 打印详细分析 ───────────────────────────────────────────────────────────
    ok_results = {k: v for k, v in results.items() if v["status"] == "OK" and v["accuracy"] is not None}
    if len(ok_results) >= 2:
        best_method = max(ok_results.keys(), key=lambda k: ok_results[k]["accuracy"])
        worst_method = min(ok_results.keys(), key=lambda k: ok_results[k]["accuracy"])
        print(f"\n结论：")
        print(f"  最佳：{best_method}（准确率 {ok_results[best_method]['accuracy']:.4f}）")
        print(f"  最差：{worst_method}（准确率 {ok_results[worst_method]['accuracy']:.4f}）")

        # 计算相对提升
        zs_key = "LLM zero-shot"
        if zs_key in ok_results:
            zs_acc = ok_results[zs_key]["accuracy"]
            for method, r in ok_results.items():
                if method != zs_key and r["accuracy"] > zs_acc:
                    diff = r["accuracy"] - zs_acc
                    print(f"  {method} 相比 zero-shot 提升：+{diff:.4f}（{diff/zs_acc*100:.1f}%）")

    # ── 各类别准确率细表 ───────────────────────────────────────────────────────
    if args.analysis:
        print(f"\n{'='*65}")
        print("各类别准确率分析")
        print(f"{'='*65}")
        header = f"{'类别':<8}"
        for method in results:
            header += f" {method[:10]:>10}"
        print(header)
        print("-" * 65)

        # 收集所有出现过的类别
        all_classes = set()
        for r in results.values():
            all_classes.update(r.get("per_class", {}).keys())

        for cls in sorted(all_classes, key=lambda c: LABEL_NAMES.index(c) if c in LABEL_NAMES else 99):
            row = f"{cls:<8}"
            for method, r in results.items():
                pc = r.get("per_class", {}).get(cls)
                if pc and pc["total"] > 0:
                    acc = pc["correct"] / pc["total"]
                    row += f" {acc:>10.2f}"
                else:
                    row += f" {'-':>10}"
            print(row)

        # 每个方法打印 top-3 / bottom-3 类别
        for method, r in results.items():
            if r["status"] != "OK" or not r.get("per_class"):
                continue
            pc = r["per_class"]
            sorted_cls = sorted(pc.items(), key=lambda x: x[1]["correct"] / max(x[1]["total"], 1))
            print(f"\n{method} 表现最好的类别：")
            for name, stats in sorted_cls[-3:][::-1]:
                acc = stats["correct"] / max(stats["total"], 1)
                print(f"  {name}: {acc:.2f} ({stats['correct']}/{stats['total']})")
            print(f"  表现最差的类别：")
            for name, stats in sorted_cls[:3]:
                acc = stats["correct"] / max(stats["total"], 1)
                print(f"  {name}: {acc:.2f} ({stats['correct']}/{stats['total']})")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "four_way_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 序列化时跳过大型 results 列表（只保留统计值）
    save_dict = {}
    for k, v in results.items():
        save_dict[k] = {kk: vv for kk, vv in v.items() if kk != "status" or v["status"] == "OK"}
        save_dict[k]["status"] = v["status"]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
