import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import CFG, ROOT, output_dir


def plot(dataset):
    target = output_dir(dataset)
    figures = ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    pair = pd.read_csv(target / "pair_metrics.csv")
    view = pair.melt(id_vars=["dataset", "method"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=view, x="method", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"{dataset}: pair-classification metrics on full test set")
    fig.tight_layout()
    fig.savefig(figures / f"{dataset}_pair_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    bad = pd.read_csv(target / "badcases.csv")
    counts = bad.groupby(["method", "error"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=counts, x="method", y="count", hue="error", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"{dataset}: false-positive and false-negative counts")
    fig.tight_layout()
    fig.savefig(figures / f"{dataset}_badcase_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    times = []
    for name in ("bi_cosine", "bi_triplet", "cross", "llm_lora"):
        meta = json.loads((target / name / "meta.json").read_text(encoding="utf-8"))
        times.append({"model": name, "seconds": meta["seconds"]})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=pd.DataFrame(times), x="model", y="seconds", ax=ax, color="#4C78A8")
    ax.axhline(300, color="#E45756", linestyle="--")
    ax.set_title(f"{dataset}: training time")
    fig.tight_layout()
    fig.savefig(figures / f"{dataset}_training_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *CFG["datasets"]], default="all")
    args = parser.parse_args()
    datasets = list(CFG["datasets"]) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        plot(dataset)


if __name__ == "__main__":
    main()
