"""
人民日报 NER 数据集探索性数据分析与可视化
"""

import os
import json
import argparse
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────── 路径与映射 ────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "peoples_daily")
FIG_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_CN = {"PER": "人名", "LOC": "地名", "ORG": "机构名"}

# ──────────────────────────── argparse ──────────────────────────────
parser = argparse.ArgumentParser(description="人民日报 NER 数据集探索性数据分析")
args = parser.parse_args()

# ──────────────────────────── 读取数据 ──────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_json(os.path.join(DATA_DIR, "train.json"))
val_data   = load_json(os.path.join(DATA_DIR, "validation.json"))
all_data   = train_data + val_data

print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")
print(f"总样本数:     {len(all_data)}")
print()

# ──────────────────────────── 从 ner_tags 提取实体 ─────────────────
def extract_entities(tokens, ner_tags):
    """从 BIO 标签序列中提取 (entity_type, start, end, text)"""
    entities = []
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                i += 1
            end = i - 1
            text = "".join(tokens[start:end + 1])
            entities.append((etype, start, end, text))
        else:
            i += 1
    return entities

# ──────────────────────────── 统计 ──────────────────────────────────
entity_type_counter = Counter()
entity_lengths = []
text_lengths = []
entities_per_sent = []
entity_examples = {}  # type -> list of text

for sample in all_data:
    tokens = sample["tokens"]
    ner_tags = sample["ner_tags"]
    text_lengths.append(len(tokens))

    ents = extract_entities(tokens, ner_tags)
    entities_per_sent.append(len(ents))

    for etype, start, end, text in ents:
        entity_type_counter[etype] += 1
        entity_lengths.append(end - start + 1)
        if etype not in entity_examples:
            entity_examples[etype] = []
        entity_examples[etype].append(text)

# ─────── 1. 各实体类型的频次计数 ───────
print("=" * 50)
print("1. 各实体类型频次计数（降序）")
print("=" * 50)
for etype, cnt in entity_type_counter.most_common():
    cn = LABEL_CN.get(etype, etype)
    print(f"  {etype:<6s}({cn})  {cnt:>8d}")
print(f"  {'总计':<10s}       {sum(entity_type_counter.values()):>8d}")
print()

# ─────── 2. 文本长度分布 ───────
print("=" * 50)
print("2. 文本长度分布（字符数）")
print("=" * 50)
import numpy as np
text_len_arr = np.array(text_lengths)
print(f"  最小值: {text_len_arr.min():>6d}")
print(f"  最大值: {text_len_arr.max():>6d}")
print(f"  平均值: {text_len_arr.mean():>8.1f}")
print(f"  中位数: {np.median(text_len_arr):>8.1f}")
print(f"  P95:    {np.percentile(text_len_arr, 95):>8.1f}")
print(f"  P99:    {np.percentile(text_len_arr, 99):>8.1f}")
print()

# ─────── 3. 实体长度分布 ───────
print("=" * 50)
print("3. 实体长度分布（end - start + 1）")
print("=" * 50)
ent_len_arr = np.array(entity_lengths)
ent_len_counter = Counter(entity_lengths)
print(f"  最小值: {ent_len_arr.min():>6d}")
print(f"  最大值: {ent_len_arr.max():>6d}")
print(f"  平均值: {ent_len_arr.mean():>8.1f}")
print(f"  中位数: {np.median(ent_len_arr):>8.1f}")
for length in sorted(ent_len_counter.keys())[:10]:
    print(f"  长度 {length:>2d}: {ent_len_counter[length]:>6d} 个实体")
print()

# ─────── 4. 每句实体数量 ───────
print("=" * 50)
print("4. 每句实体数量")
print("=" * 50)
eps_arr = np.array(entities_per_sent)
print(f"  最小值: {eps_arr.min():>6d}")
print(f"  最大值: {eps_arr.max():>6d}")
print(f"  平均值: {eps_arr.mean():>8.2f}")
print(f"  无实体句子: {int((eps_arr == 0).sum()):>6d} ({(eps_arr == 0).mean() * 100:.1f}%)")
print()

# ─────── 5. 各类型去重示例 ───────
print("=" * 50)
print("5. 各类型去重示例（前5个）")
print("=" * 50)
for etype in sorted(entity_examples.keys()):
    cn = LABEL_CN.get(etype, etype)
    unique = list(dict.fromkeys(entity_examples[etype]))[:5]
    print(f"  {etype}({cn}): {unique}")
print()

# ══════════════════════════════ 可视化 ══════════════════════════════

# ─────── 图1: 实体频次柱状图 ───────
fig1, ax1 = plt.subplots(figsize=(8, 5))
etypes_sorted = [t for t, _ in entity_type_counter.most_common()]
counts_sorted = [entity_type_counter[t] for t in etypes_sorted]
x_labels = [f"{t}\n({LABEL_CN.get(t, t)})" for t in etypes_sorted]
bars = ax1.bar(x_labels, counts_sorted, color="#4C72B0")
for bar, cnt in zip(bars, counts_sorted):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts_sorted) * 0.01,
             str(cnt), ha="center", va="bottom", fontsize=11)
ax1.set_title("实体类型频次分布")
ax1.set_ylabel("频次")
ax1.set_xlabel("实体类型")
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "entity_freq.png"), dpi=120)
plt.close(fig1)
print(f"[图1] 实体频次柱状图已保存 → outputs/figures/entity_freq.png")

# ─────── 图2: 文本长度直方图 ───────
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(text_lengths, bins=80, color="#4C72B0", alpha=0.7, edgecolor="white")
p95_val = int(np.percentile(text_len_arr, 95))
ax2.axvline(64, color="red",   linestyle="--", linewidth=1.5, label="max_length=64")
ax2.axvline(128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
ax2.axvline(p95_val, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95_val}")
ax2.set_title("文本长度分布")
ax2.set_xlabel("文本长度（字符数）")
ax2.set_ylabel("样本数")
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "text_length.png"), dpi=120)
plt.close(fig2)
suggested_max_length = p95_val
print(f"[图2] 文本长度直方图已保存 → outputs/figures/text_length.png")
print(f"  建议max_length: {suggested_max_length}")

# ─────── 图3: 实体长度柱状图（前20种长度） ───────
fig3, ax3 = plt.subplots(figsize=(10, 5))
top20_lengths = sorted(ent_len_counter.keys())[:20]
top20_counts  = [ent_len_counter[l] for l in top20_lengths]
bars3 = ax3.bar([str(l) for l in top20_lengths], top20_counts, color="#55A868")
for bar, cnt in zip(bars3, top20_counts):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(top20_counts) * 0.01,
             str(cnt), ha="center", va="bottom", fontsize=9)
ax3.set_title("实体长度分布（前20种长度）")
ax3.set_xlabel("实体长度（字符数）")
ax3.set_ylabel("实体数量")
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "entity_length.png"), dpi=120)
plt.close(fig3)
print(f"[图3] 实体长度柱状图已保存 → outputs/figures/entity_length.png")
print("  提示: CRF 对短实体（长度1~3）有显著优势，能通过转移约束减少碎片化错误")

# ════════════════════════════ 下一步引导 ════════════════════════════
print()
print("=" * 50)
print("下一步:")
print("  python train.py")
print("  python train.py --use_crf")
print("=" * 50)
