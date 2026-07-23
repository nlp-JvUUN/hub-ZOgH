"""
数据集探索与可视化 — 支持 cluener2020 和 peoples_daily

教学重点：
  1. span/BIO 标注格式的统计方法
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python explore_data.py                        # CLUENER（默认）
  python explore_data.py --dataset peoples_daily  # 人民日报

输出：
  outputs/figures/{dataset}_entity_distribution.png
  outputs/figures/{dataset}_text_length_distribution.png
  outputs/figures/{dataset}_entity_length_distribution.png
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

from dataset_config import get_config

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "outputs" / "figures"


def load_split(split: str, data_dir: Path) -> list:
    path = data_dir / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats_span(records: list) -> dict:
    """CLUENER span 格式的统计。"""
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        text = row["text"]
        text_lengths.append(len(text))
        label = row.get("label") or {}

        total_entities = 0
        for etype, spans in label.items():
            for surface, positions in spans.items():
                for start, end in positions:
                    entity_type_counts[etype] += 1
                    entity_lengths.append(end - start + 1)
                    total_entities += 1
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(surface)

        entity_per_sentence.append(total_entities)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def collect_stats_bio(records: list) -> dict:
    """peoples_daily BIO token-tag 格式的统计。"""
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        text = "".join(tokens)
        text_lengths.append(len(text))

        total_entities = 0
        i = 0
        while i < len(ner_tags):
            tag = ner_tags[i]
            if tag.startswith("B-"):
                etype = tag[2:]
                start = i
                i += 1
                while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                    i += 1
                end = i
                surface = "".join(tokens[start:end])
                entity_type_counts[etype] += 1
                entity_lengths.append(end - start)
                total_entities += 1
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append(surface)
            else:
                i += 1

        entity_per_sentence.append(total_entities)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict, dataset: str, entity_types_zh: dict):
    print("=" * 70)
    print(f"{dataset} 数据集统计摘要")
    print("=" * 70)

    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    print(f"  文本平均长度：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} 字")
    print(f"  文本最大长度：{max(stats_train['text_lengths'])} 字")
    print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths'])//2]} 字")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    if stats_train['entity_lengths']:
        print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    print("\n【各类实体频次（训练集）】")
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = entity_types_zh.get(etype, etype)
        print(f"  {etype:15s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = entity_types_zh.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:15s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict, dataset: str, entity_types_zh: dict):
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({entity_types_zh.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{dataset} 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{dataset}_entity_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'{dataset}_entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict, dataset: str):
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title(f"{dataset} 文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{dataset}_text_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'{dataset}_text_length_distribution.png'}")
    plt.close()
    print(f"  P95 文本长度={p95}")


def plot_entity_length_distribution(stats_train: dict, dataset: str):
    lengths = Counter(stats_train["entity_lengths"])
    if not lengths:
        print("  无实体数据，跳过实体长度分布图")
        return
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title(f"{dataset} 实体长度分布（训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{dataset}_entity_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'{dataset}_entity_length_distribution.png'}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字")


def main():
    args = parse_args()
    cfg = get_config(args.dataset)
    data_dir = Path(cfg["data_dir"])
    fmt = cfg["format"]
    entity_types_zh = cfg["entity_types_zh"]

    collect_stats = collect_stats_bio if fmt == "bio" else collect_stats_span

    train_records = load_split("train", data_dir)
    val_records = load_split("validation", data_dir)

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val, args.dataset, entity_types_zh)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train, args.dataset, entity_types_zh)
    plot_text_length_distribution(stats_train, args.dataset)
    plot_entity_length_distribution(stats_train, args.dataset)

    print(f"\n探索完成！图表已保存到 outputs/figures/")
    print(f"下一步：python train.py --dataset {args.dataset}")
    print(f"         python train.py --dataset {args.dataset} --use_crf")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 NER 数据集")
    parser.add_argument("--dataset", type=str, default="cluener",
                        choices=["cluener", "peoples_daily"],
                        help="数据集名称（默认 cluener）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
