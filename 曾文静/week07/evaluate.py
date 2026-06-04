# -*- coding: utf-8 -*-
"""
week07/evaluate.py
==================
评估训练好的 BERT 序列标注模型:
  - entity-level P/R/F1 (seqeval)
  - 逐实体类型 F1
  - 非法 BIO 序列统计 (量化 CRF 的价值)
  - 推理 demo: 输入文本 -> 抽取实体
  - --compare: 同时评估 Linear 与 CRF 两个模型并对比

用法:
  python evaluate.py                        # 评估 BERT+Linear
  python evaluate.py --use_crf              # 评估 BERT+CRF
  python evaluate.py --compare              # 两模型对比
  python evaluate.py --ckpt outputs/checkpoints/best_crf.pt
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dataset import build_dataloaders, build_label_schema
from metrics import entity_f1, illegal_bio_stats, per_type_report, recover_entities
from model import NERModel
from train import predict_tags


def load_model(ckpt_path, bert_path, num_labels, device):
    """加载 checkpoint, 返回 (model, use_crf)"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = NERModel(bert_path, num_labels=num_labels,
                     use_crf=ckpt.get("use_crf", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("use_crf", False)


def evaluate_split(model, loader, id2label, device, split="validation"):
    """在指定 split 上评估, 返回指标 + 预测"""
    gold, pred = predict_tags(model, loader, loader.dataset, id2label, device)
    metrics = entity_f1(gold, pred)
    illegal = illegal_bio_stats(pred)
    print(f"\n[{split} 集] entity-level P/R/F1: "
          f"{metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
    print(f"[{split} 集] 非法BIO序列: {illegal}")
    return metrics, gold, pred, illegal


def demo_inference(model, tokenizer, id2label, device, texts, max_length=128):
    """推理演示: 输入句子 -> 打印识别出的实体"""
    print("\n" + "=" * 60)
    print("推理 Demo")
    print("=" * 60)
    for text in texts:
        chars = list(text)
        enc = tokenizer(chars, is_split_into_words=True, truncation=True,
                        padding="max_length", max_length=max_length,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc["token_type_ids"].to(device)
        with torch.no_grad():
            _, decoded = model(input_ids, attention_mask, token_type_ids)

        seq = decoded[0] if model.use_crf else decoded[0].tolist()
        pred_tags, prev_wid = [], None
        for j, wid in enumerate(enc.word_ids()):
            if wid is None or wid == prev_wid or j >= len(seq):
                continue
            pred_tags.append(id2label[seq[j]])
            prev_wid = wid

        entities = recover_entities(chars[:len(pred_tags)], pred_tags)
        print(f"\n文本: {text}")
        print(f"标签: {' '.join(pred_tags)}")
        if entities:
            for t, s, e, txt in entities:
                print(f"  [{t}] {txt} (位置 {s}-{e})")
        else:
            print("  未识别出实体")


def main():
    parser = argparse.ArgumentParser(description="评估 BERT 序列标注模型")
    parser.add_argument("--use_crf", action="store_true", help="评估CRF模型")
    parser.add_argument("--compare", action="store_true",
                        help="同时评估Linear与CRF并对比")
    parser.add_argument("--ckpt", type=str, default=None, help="指定checkpoint")
    parser.add_argument("--bert_path", type=str, default="bert-base-chinese")
    parser.add_argument("--data_dir", type=str, default="data/cluener")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, label2id, id2label = build_label_schema()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=True)
    loaders = build_dataloaders(args.data_dir, tokenizer, label2id,
                                args.batch_size, args.max_length)
    loader = loaders[args.split]

    checkpoints = Path(args.output_dir) / "checkpoints"
    if args.ckpt:
        paths = [args.ckpt]
    elif args.compare:
        paths = [checkpoints / "best_linear.pt", checkpoints / "best_crf.pt"]
    elif args.use_crf:
        paths = [checkpoints / "best_crf.pt"]
    else:
        paths = [checkpoints / "best_linear.pt"]

    results = {}
    for path in paths:
        name = "BERT+CRF" if "crf" in str(path) else "BERT+Linear"
        print(f"\n{'='*60}\n评估: {name} ({path})\n{'='*60}")
        model, use_crf = load_model(path, args.bert_path, len(labels), device)
        metrics, gold, pred, illegal = evaluate_split(model, loader, id2label,
                                                      device, args.split)
        results[name] = {"metrics": metrics, "illegal": illegal}
        print("\n逐实体类型报告:")
        print(per_type_report(gold, pred))

    if args.compare and len(results) > 1:
        print("\n" + "=" * 60)
        print("两模型对比")
        print("=" * 60)
        print(f"{'方案':<14}{'Precision':>12}{'Recall':>12}{'F1':>12}{'非法序列':>10}")
        print("-" * 60)
        for name, r in results.items():
            m, il = r["metrics"], r["illegal"]
            total = il["非法I开头序列"] + il["非法跨类型转移"]
            print(f"{name:<14}{m['precision']:>12.4f}{m['recall']:>12.4f}"
                  f"{m['f1']:>12.4f}{total:>10}")

    # 保存评估结果
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        fname = "compare"
    elif args.use_crf or (args.ckpt and "crf" in args.ckpt):
        fname = "crf"
    else:
        fname = "linear"
    eval_path = log_dir / f"eval_{fname}_{args.split}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 评估结果 -> {eval_path}")

    # 推理 demo
    model, _ = load_model(paths[0], args.bert_path, len(labels), device)
    demo_inference(model, tokenizer, id2label, device, [
        "华为技术有限公司总裁任正非在深圳总部接受采访",
        "北京大学的李华教授参加了在上海举办的人工智能大会",
        "今天天气不错，适合出去散步",
    ])


if __name__ == "__main__":
    main()
