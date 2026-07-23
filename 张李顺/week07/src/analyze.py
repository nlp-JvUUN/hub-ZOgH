import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import CFG, ROOT, load_rows, output_dir


def summarize(dataset):
    records = []
    splits = {}
    for split in ("train", "validation", "test"):
        rows = load_rows(dataset, split)
        splits[split] = rows
        labels = np.array([int(x["label"]) for x in rows])
        l1 = np.array([len(x["sentence1"]) for x in rows])
        l2 = np.array([len(x["sentence2"]) for x in rows])
        pairs = [(x["sentence1"], x["sentence2"]) for x in rows]
        records.append({
            "dataset": dataset,
            "split": split,
            "rows": len(rows),
            "positive_rate": labels.mean(),
            "sentence1_mean_len": l1.mean(),
            "sentence2_mean_len": l2.mean(),
            "sentence1_p95_len": np.percentile(l1, 95),
            "sentence2_p95_len": np.percentile(l2, 95),
            "sentence1_gt80": int((l1 > 80).sum()),
            "sentence2_gt80": int((l2 > 80).sum()),
            "duplicate_pair_rate": 1 - len(set(pairs)) / len(pairs),
            "identical_text_rate": np.mean([a == b for a, b in pairs])
        })
    return pd.DataFrame(records), splits


def plot_dataset(dataset, splits):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = list(splits)
    sizes = [len(splits[x]) for x in names]
    rates = [np.mean([r["label"] for r in splits[x]]) for x in names]
    axes[0, 0].bar(names, sizes, color="#4C78A8")
    axes[0, 0].set_title("Split size")
    axes[0, 1].bar(names, rates, color="#F58518")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Positive-label rate")
    train = splits["train"]
    l1 = np.array([len(x["sentence1"]) for x in train])
    l2 = np.array([len(x["sentence2"]) for x in train])
    normal1 = l1[l1 <= 80]
    normal2 = l2[l2 <= 80]
    bins = np.arange(0, 82, 2)
    axes[1, 0].hist(normal1, bins=bins, weights=np.full(len(normal1), 100 / len(normal1)), histtype="step", linewidth=2.2, label="Text A", color="#54A24B")
    axes[1, 0].hist(normal2, bins=bins, weights=np.full(len(normal2), 100 / len(normal2)), histtype="step", linewidth=2.2, label="Text B", color="#E45756")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 80)
    axes[1, 0].set_xlabel("Character count")
    axes[1, 0].set_ylabel("Samples per 2-char bin (%)")
    axes[1, 0].set_title("Train text-length distribution")
    axes[1, 0].text(.98, .72, f"Excluded >80: A={int((l1 > 80).sum())}, B={int((l2 > 80).sum())}", transform=axes[1, 0].transAxes, ha="right")
    table = []
    for name in names:
        labels = [int(x["label"]) for x in splits[name]]
        table.append([labels.count(0), labels.count(1)])
    bottom = np.zeros(len(names))
    for i, label in enumerate(("negative", "positive")):
        values = [x[i] for x in table]
        axes[1, 1].bar(names, values, bottom=bottom, label=label)
        bottom += values
    axes[1, 1].legend()
    axes[1, 1].set_title("Label counts")
    fig.suptitle(dataset)
    fig.tight_layout()
    path = ROOT / "outputs" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{dataset}_data_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(datasets):
    frames = []
    for dataset in datasets:
        frame, splits = summarize(dataset)
        frames.append(frame)
        plot_dataset(dataset, splits)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pd.concat(frames).to_csv(out / "data_summary.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *CFG["datasets"]], default="all")
    args = parser.parse_args()
    run(list(CFG["datasets"]) if args.dataset == "all" else [args.dataset])


if __name__ == "__main__":
    main()
