"""
NLP 推理演示：输入一句话，输出识别到的实体

使用方式：
  python inference.py                          # 交互式模式，输入句子后回车
  python inference.py --text "张三在北京工作"   # 单次推理
  python inference.py --use_crf                # 使用 CRF 模型（如果训练过）

依赖：
  python train.py 或 python train.py --use_crf  （先训练好模型）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
from transformers import BertTokenizer

from dataset import build_label_schema
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs_peoples_daily" / "checkpoints"


def predict(
    model,
    tokenizer,
    text: str,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> list[dict]:
    """输入文本，返回识别到的实体列表。
    返回格式：[{"text": "张三", "type": "PER", "start": 0, "end": 1}, ...]
    """
    max_length = 128
    chars = list(text)

    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        if use_crf:
            pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            pred_ids = pred_ids_list[0]
        else:
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            pred_ids = logits.argmax(dim=-1)[0].tolist()

    # 解码：跳过 [CLS](-100) 和 [SEP]/[PAD](-100)，只取有效 token
    word_ids = encoding.word_ids(batch_index=0)
    bio_labels = []
    for j, wid in enumerate(word_ids):
        if wid is None:
            continue
        if j >= len(pred_ids):
            break
        bio_labels.append(id2label[pred_ids[j]])

    # BIO 解码 → 提取实体 span
    entities = []
    i = 0
    while i < len(bio_labels):
        label = bio_labels[i]
        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            end = i
            j = i + 1
            while j < len(bio_labels) and bio_labels[j] == f"I-{entity_type}":
                end = j
                j += 1
            entity_text = "".join(chars[start:end + 1])
            entities.append({
                "text": entity_text,
                "type": entity_type,
                "start": start,
                "end": end,
            })
            i = j
        else:
            i += 1

    return entities


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"[错误] 找不到模型：{ckpt_path}")
        print(f"请先运行：python train.py {'--use_crf' if args.use_crf else ''}")
        return

    # 加载模型
    labels, label2id, id2label = build_label_schema()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=len(labels),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    print(f"模型已加载：{'BERT+CRF' if args.use_crf else 'BERT+Linear'}")
    print(f"标签体系：{labels}")
    print(f"实体类型：PER(人名) / ORG(机构) / LOC(地点)\n")

    # 交互模式 or 单次推理
    if args.text:
        texts = [args.text]
    else:
        print("输入文本进行 NER 识别（输入 'quit' 退出）：\n")
        texts = []
        while True:
            t = input("> ").strip()
            if t.lower() in ("quit", "exit", "q"):
                break
            if t:
                texts.append(t)

    for text in texts:
        entities = predict(model, tokenizer, text, id2label, device, args.use_crf)
        print(f"\n输入：「{text}」")
        if entities:
            print("识别到的实体：")
            for e in entities:
                type_cn = {"PER": "人名", "ORG": "机构", "LOC": "地点"}.get(e["type"], e["type"])
                print(f"  [{e['start']}:{e['end']}] \"{e['text']}\" → {e['type']}({type_cn})")
        else:
            print("  未识别到实体")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="NER 推理演示")
    parser.add_argument("--text", type=str, help="输入文本（不指定则进入交互模式）")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
