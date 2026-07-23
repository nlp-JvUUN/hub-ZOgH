"""
数据集探索与可视化

支持 cluener2020（span格式）和 peoples_daily（CoNLL格式）。

教学重点：
  1. span 标注格式的统计方法（与 BIO 格式的信息等价）
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python explore_data.py                           # cluener（默认）
  python explore_data.py --dataset peoples_daily   # 人民日报 NER

输出：
  outputs/figures/entity_distribution.png          各类实体频次分布
  outputs/figures/text_length_distribution.png     文本长度分布
  outputs/figures/entity_length_distribution.png   实体长度分布
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

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "outputs" / "figures"

# 中文标签对照
ENTITY_LABELS = {
    "cluener": {
        "address": "地址", "book": "书名", "company": "公司",
        "game": "游戏", "government": "政府", "movie": "影视",
        "name": "人名", "organization": "组织", "position": "职位",
        "scene": "景点",
    },
    "peoples_daily": {
        "PER": "人名", "ORG": "组织", "LOC": "地点",
    },
}

def load_split(split: str, data_dir: Path) -> list:
    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(records: list) -> dict:
    """收集 cluener2020 格式（span标注）的统计信息。"""
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


def collect_stats_peoples_daily(records: list) -> dict:
    """收集人民日报 NER 格式（CoNLL）的统计信息。"""
    from collections import Counter
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        text_lengths.append(len(tokens))

        total_entities = 0
        current_etype = None
        current_start = None

        for i, tag in enumerate(ner_tags):
            if tag.startswith("B-"):
                # 结束上一个实体
                if current_etype is not None:
                    entity_lengths.append(i - current_start)
                current_etype = tag[2:]
                current_start = i
                total_entities += 1
            elif tag.startswith("I-"):
                # 延续当前实体
                pass
            else:  # O
                # 结束上一个实体
                if current_etype is not None:
                    entity_lengths.append(i - current_start)
                    current_etype = None
                    current_start = None

        # 处理序列末尾的实体
        if current_etype is not None:
            entity_lengths.append(len(ner_tags) - current_start)

        # 统计实体类型频次
        for tag in ner_tags:
            if tag.startswith("B-"):
                etype = tag[2:]
                entity_type_counts[etype] += 1

        # 收集每类实体的示例文本
        current_etype = None
        current_start = None
        for i, tag in enumerate(ner_tags):
            if tag.startswith("B-"):
                if current_etype is not None:
                    surface = "".join(tokens[current_start:i])
                    if current_etype not in entities_by_type:
                        entities_by_type[current_etype] = []
                    entities_by_type[current_etype].append(surface)
                current_etype = tag[2:]
                current_start = i
            elif tag == "O" and current_etype is not None:
                surface = "".join(tokens[current_start:i])
                if current_etype not in entities_by_type:
                    entities_by_type[current_etype] = []
                entities_by_type[current_etype].append(surface)
                current_etype = None
                current_start = None
        # 末尾实体
        if current_etype is not None:
            surface = "".join(tokens[current_start:])
            if current_etype not in entities_by_type:
                entities_by_type[current_etype] = []
            entities_by_type[current_etype].append(surface)

        entity_per_sentence.append(total_entities)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict, dataset_name: str):
    et_label = ENTITY_LABELS[dataset_name]

    print("=" * 70)
    print(f"{'cluener2020' if dataset_name == 'cluener' else '人民日报 NER'} 数据集统计摘要")
    print("=" * 70)

    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    print(f"  文本平均长度：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} 字")
    print(f"  文本最大长度：{max(stats_train['text_lengths'])} 字")
    print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths'])//2]} 字")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    print("\n【各类实体频次（训练集）】")
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        print(f"  {etype:15s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:15s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict, dataset_name: str):
    et_label = ENTITY_LABELS[dataset_name]
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k,k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    dataset_title = "cluener2020" if dataset_name == "cluener" else "人民日报 NER"
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{dataset_title} 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{dataset_name}" if dataset_name != "cluener" else ""
    out_path = FIG_DIR / f"entity_distribution{suffix}.png"
    fig.savefig(out_path, dpi=120)
    print(f"  已保存 → {out_path}")
    plt.close()


def plot_text_length_distribution(stats_train: dict, dataset_name: str):
    lengths = stats_train["text_lengths"]
    dataset_title = "cluener2020" if dataset_name == "cluener" else "人民日报 NER"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title(f"文本长度分布（{dataset_title}，训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{dataset_name}" if dataset_name != "cluener" else ""
    out_path = FIG_DIR / f"text_length_distribution{suffix}.png"
    fig.savefig(out_path, dpi=120)
    print(f"  已保存 → {out_path}")
    plt.close()
    print(f"  P95 文本长度={p95}，建议 max_length=128")


def plot_entity_length_distribution(stats_train: dict, dataset_name: str):
    from collections import Counter
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    dataset_title = "cluener2020" if dataset_name == "cluener" else "人民日报 NER"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title(f"实体长度分布（{dataset_title}，训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{dataset_name}" if dataset_name != "cluener" else ""
    out_path = FIG_DIR / f"entity_length_distribution{suffix}.png"
    fig.savefig(out_path, dpi=120)
    print(f"  已保存 → {out_path}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字，CRF 对短实体边界识别优势更明显")


def main():
    args = parse_args()
    dataset_name = args.dataset

    data_dir = ROOT / "data" / dataset_name
    train_records = load_split("train", data_dir)
    val_records = load_split("validation", data_dir)

    if dataset_name == "peoples_daily":
        stats_train = collect_stats_peoples_daily(train_records)
        stats_val = collect_stats_peoples_daily(val_records)
    else:
        stats_train = collect_stats(train_records)
        stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val, dataset_name)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train, dataset_name)
    plot_text_length_distribution(stats_train, dataset_name)
    plot_entity_length_distribution(stats_train, dataset_name)

    print("\n探索完成！图表已保存到 outputs/figures/")
    dataset_flag = f" --dataset {dataset_name}" if dataset_name != "cluener" else ""
    print(f"下一步：python train.py{dataset_flag}               # 训练 BERT+Linear")
    print(f"         python train.py{dataset_flag} --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 NER 数据集")
    parser.add_argument("--dataset", type=str, default="cluener", choices=["cluener", "peoples_daily"],
                        help="数据集名称")
    return parser.parse_args()


if __name__ == "__main__":
    main()
