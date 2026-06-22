"""
NER 数据集探索与可视化 — 支持 cluener 和 peoples_daily

教学重点：
  1. span vs BIO 两种标注格式的统计方法
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python explore_data.py --dataset cluener
  python explore_data.py --dataset peoples_daily

输出：
  outputs/figures/entity_distribution_{dataset}.png   各类实体频次分布
  outputs/figures/text_length_distribution_{dataset}.png  文本长度分布
  outputs/figures/entity_length_distribution_{dataset}.png 实体长度分布
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
ORIGINAL_DATA = Path(
    "D:/aipy/AI大模型培训部分/week7序列标注问题_0530/"
    "week7 序列标注问题/序列标注项目/data"
)
FIG_DIR = ROOT / "outputs" / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════════════

ENTITY_LABELS_CLUENER = {
    "address": "地址", "book": "书名", "company": "公司",
    "game": "游戏", "government": "政府机构", "movie": "影视作品",
    "name": "人名", "organization": "组织机构", "position": "职位",
    "scene": "景点/场所",
}

ENTITY_LABELS_PD = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_data(dataset: str) -> tuple[list, list]:
    """加载训练集和验证集。"""
    data_dir = ORIGINAL_DATA / dataset
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
    return train_data, val_data


# ══════════════════════════════════════════════════════════════════════════════
# 统计（按数据格式区分）
# ══════════════════════════════════════════════════════════════════════════════

def collect_stats_cluener(records: list) -> dict:
    """cluener span 格式统计。"""
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
    """peoples_daily BIO 格式统计。"""
    from dataset import bio_tags_to_entities
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        tokens = row["tokens"]
        text_lengths.append(len(tokens))
        ner_tags = row["ner_tags"]

        entities = bio_tags_to_entities(tokens, ner_tags)
        total_entities = len(entities)

        for ent in entities:
            etype = ent["type"]
            surface = ent["text"]
            entity_type_counts[etype] += 1
            entity_lengths.append(len(surface))
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


# ══════════════════════════════════════════════════════════════════════════════
# 打印摘要
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(stats_train: dict, stats_val: dict, dataset: str, et_label: dict):
    print("=" * 70)
    print(f"{dataset} 数据集统计摘要")
    print("=" * 70)

    n_train = len(stats_train['text_lengths'])
    n_val = len(stats_val['text_lengths'])

    print(f"\n【样本数】")
    print(f"  训练集：{n_train} 条")
    print(f"  验证集：{n_val} 条")

    print(f"\n【训练集文本长度】")
    print(f"  平均长度：{sum(stats_train['text_lengths']) / n_train:.1f} 字")
    print(f"  最大长度：{max(stats_train['text_lengths'])} 字")
    print(f"  中位数：{sorted(stats_train['text_lengths'])[n_train // 2]} 字")
    p95 = sorted(stats_train['text_lengths'])[int(n_train * 0.95)]
    print(f"  P95：{p95} 字")

    n_entities = sum(stats_train['entity_type_counts'].values())
    print(f"\n【训练集实体】")
    print(f"  实体总数：{n_entities}")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / n_train:.2f}")
    if stats_train['entity_lengths']:
        print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    # 含实体句子比例
    has_entity = sum(1 for n in stats_train['entity_per_sentence'] if n > 0)
    print(f"  含实体句子比例：{has_entity}/{n_train}（{has_entity/n_train*100:.1f}%）")

    print(f"\n【各类实体频次（训练集）】")
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        print(f"  {etype:10s} ({cn:6s}) : {cnt:6d} 条")

    print(f"\n【各类实体示例（训练集，前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:10s} ({cn}) : {' | '.join(examples)}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_entity_distribution(stats_train: dict, dataset: str, et_label: dict):
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values)*0.01,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{dataset} 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"entity_distribution_{dataset}.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'entity_distribution_{dataset}.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict, dataset: str):
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title(f"文本长度分布（训练集）— {dataset}", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"text_length_distribution_{dataset}.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'text_length_distribution_{dataset}.png'}")
    plt.close()
    print(f"  P95 文本长度={p95}，建议 max_length=128")


def plot_entity_length_distribution(stats_train: dict, dataset: str):
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title(f"实体长度分布（训练集，前20）— {dataset}", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"entity_length_distribution_{dataset}.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / f'entity_length_distribution_{dataset}.png'}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    ds = args.dataset

    if ds == "peoples_daily":
        et_label = ENTITY_LABELS_PD
        collect_fn = collect_stats_peoples_daily
    else:
        et_label = ENTITY_LABELS_CLUENER
        collect_fn = collect_stats_cluener

    train_records, val_records = load_data(ds)

    stats_train = collect_fn(train_records)
    stats_val = collect_fn(val_records)

    print_summary(stats_train, stats_val, ds, et_label)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train, ds, et_label)
    plot_text_length_distribution(stats_train, ds)
    plot_entity_length_distribution(stats_train, ds)

    print(f"\n探索完成！图表已保存到 outputs/figures/")
    print(f"下一步：python src/train.py --dataset {ds}")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 NER 数据集")
    parser.add_argument("--dataset", type=str, choices=["cluener", "peoples_daily"],
                        default="peoples_daily", help="数据集选择")
    return parser.parse_args()


if __name__ == "__main__":
    main()
