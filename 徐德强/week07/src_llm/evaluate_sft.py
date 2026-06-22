"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估 NER entity-level F1，
与 BERT+CRF 和 LLM API（zero/few-shot）多方对比

教学重点：
  1. 生成式 NER 的评估方式：JSON 解析 → span-level F1（与 llm_ner.py 一致）
  2. LoRA adapter 自动识别：目录含 adapter_config.json → LoRA，否则 → 全量
  3. 与 BERT+CRF 的对比：生成式 vs 序列标注，各有什么优劣
  4. --dataset 参数：支持 cluener 和 peoples_daily

使用方式：
  python evaluate_sft.py --dataset peoples_daily
  python evaluate_sft.py --dataset peoples_daily --n_samples 50 --demo

依赖：
  pip install torch transformers peft
"""

import os
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel
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
LOG_DIR = ROOT / "outputs" / "logs"

# 添加 src 路径以使用 dataset 工具
sys.path.insert(0, str(ROOT / "src"))

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
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer  = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 推理与解析
# ══════════════════════════════════════════════════════════════════════════════

def generate_ner(text: str, model, tokenizer, device: torch.device,
                 system_prompt: str, max_new_tokens: int = 256) -> str:
    """生成 NER JSON 输出。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def gold_spans_from_record(record: dict, ds: str) -> set[tuple[str, str, int, int]]:
    """提取 gold spans，格式 (text, type, start, end)。"""
    if ds == "peoples_daily":
        from dataset import bio_tags_to_entities
        tokens = record["tokens"]
        ner_tags = record["ner_tags"]
        entities = bio_tags_to_entities(tokens, ner_tags)
        spans = set()
        text_str = "".join(tokens)
        for ent in entities:
            idx = text_str.find(ent["text"])
            if idx != -1:
                spans.add((ent["text"], ent["type"], idx,
                           idx + len(ent["text"]) - 1))
        return spans
    else:
        # cluener format
        spans = set()
        for etype, surfaces in (record.get("label") or {}).items():
            for surface, positions in surfaces.items():
                for start, end in positions:
                    spans.add((surface, etype, start, end))
        return spans


def pred_spans_from_output(text: str, raw_output: str,
                           entity_types: list[str]
                           ) -> set[tuple[str, str, int, int]]:
    """从 SFT 生成的 JSON 中提取 spans。"""
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return set()
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()
    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype   = str(ent.get("type", "")).strip()
        if not surface or etype not in entity_types:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level precision / recall / F1。"""
    tp         = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p  = tp / pred_total if pred_total else 0.0
    r  = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 评估")
    parser.add_argument("--dataset", type=str, choices=["cluener", "peoples_daily"],
                        default="peoples_daily", help="数据集选择")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir", default=None,
                        help="checkpoint 目录；默认 outputs/sft_adapter_{dataset}")
    parser.add_argument("--n_samples", default=100, type=int,
                        help="验证集采样数")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo", action="store_true",
                        help="只跑 5 条示例，快速演示")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ds = args.dataset

    # ── 数据集配置 ────────────────────────────────────────────────────────────
    if ds == "peoples_daily":
        data_dir = ORIGINAL_DATA / "peoples_daily"
        system_prompt = SYSTEM_PROMPT_PD
        entity_types = ENTITY_TYPES_PD
    else:
        data_dir = ORIGINAL_DATA / "cluener"
        system_prompt = SYSTEM_PROMPT_CLUENER
        entity_types = ENTITY_TYPES_CLUENER

    # ── checkpoint 目录 ──────────────────────────────────────────────────────
    if args.ckpt_dir is None:
        default_ckpt = ROOT / "outputs" / f"sft_adapter_{ds}"
        args.ckpt_dir = str(default_ckpt)
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        return

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(args.seed)
    n = 5 if args.demo else args.n_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    all_golds, all_preds = [], []
    detail_records = []
    parse_fail = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text  = record.get("text") or "".join(record["tokens"])
        g_set = gold_spans_from_record(record, ds)
        raw   = generate_ner(text, model, tokenizer, device, system_prompt)
        p_set = pred_spans_from_output(text, raw, entity_types)

        if not re.search(r"\{.*entities.*\}", raw, re.DOTALL):
            parse_fail += 1

        all_golds.append(g_set)
        all_preds.append(p_set)
        detail_records.append({
            "text": text,
            "gold":  [{"text": s, "type": t} for s, t, *_ in g_set],
            "pred":  [{"text": s, "type": t} for s, t, *_ in p_set],
            "raw_output": raw,
        })

        tp_here  = len(g_set & p_set)
        status   = "✓" if g_set == p_set else ("~" if tp_here > 0 else "✗")
        gold_str = ",".join(f"{s}({t})" for s, t, *_ in list(g_set)[:3])
        print(f"[{i:3d}/{len(samples)}] {status}  "
              f"gold:{gold_str or '无'}  |  {text[:30]}")

    elapsed = time.time() - t0
    metrics = compute_span_f1(all_golds, all_preds)

    # ── 读取已有结果做多方对比 ─────────────────────────────────────────────────
    bert_crf_f1 = "?"
    llm_zero_f1 = "?"
    llm_few_f1  = "?"
    crf_log_path = LOG_DIR / f"eval_crf_validation_{ds}.json"
    llm_log_path = LOG_DIR / f"eval_llm_{ds}.json"

    if crf_log_path.exists():
        with open(crf_log_path, encoding="utf-8") as f:
            crf_data = json.load(f)
        bert_crf_f1 = f"{crf_data.get('entity_f1', crf_data.get('f1', '?')):.4f}"

    if llm_log_path.exists():
        with open(llm_log_path, encoding="utf-8") as f:
            llm_data = json.load(f)
        llm_zero_f1 = f"{llm_data['zero_shot']['f1']:.4f}"
        llm_few_f1  = f"{llm_data['few_shot']['f1']:.4f}"

    print(f"\n{'='*65}")
    print(f"LLM SFT NER 评估结果（数据集：{ds}）")
    print(f"{'='*65}")
    print(f"  样本数      : {len(samples)}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  JSON 解析失败: {parse_fail} 条 "
          f"({parse_fail/len(samples)*100:.1f}%)")
    print(f"  总耗时      : {elapsed:.1f}s，"
          f"均值 {elapsed/len(samples):.2f}s/条")

    print(f"""
多方对比（验证集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │  F1      │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT + CRF（全量数据，3 epoch）           │ {bert_crf_f1:<8} │
  │ DeepSeek API zero-shot（{args.n_samples} 条）           │ {llm_zero_f1:<8} │
  │ DeepSeek API few-shot（{args.n_samples} 条，3 例）      │ {llm_few_f1:<8} │
  │ Qwen2.5-0.5B SFT（LoRA，{len(samples)} 条样本）  │ {metrics['f1']:.4f}   │
  └──────────────────────────────────────────┴──────────┘

评估标准说明：
  SFT（本脚本）和 LLM API（llm_ner.py）均使用 span F1：
    text.find() 近似定位 → (text, type, start, end) 4元组 → 与 gold 做集合交集
  BERT+CRF（evaluate.py）使用 seqeval：
    BIO 解码出精确 token 边界 → 严格位置匹配

思考题：
  1. SFT 本地小模型 vs LLM API few-shot，谁的 F1 更高？为什么？
  2. NER 的 JSON 输出比分类的单词输出难控制，体现在哪里（parse_fail 数）？
  3. BERT+CRF 保证零非法序列，生成式 NER 有这个保证吗？如何处理？
  4. 如果给 SFT 模型也提供 few-shot 示例（系统提示里加样例），F1 会提升吗？
""")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_path = LOG_DIR / f"eval_sft_{ds}.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {k: (v if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()},
            "n_samples": len(samples), "parse_fail": parse_fail,
            "detail": detail_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
