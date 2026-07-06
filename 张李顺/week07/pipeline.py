import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(ROOT)


def call(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def train(dataset):
    call("src.train_bi", "--dataset", dataset, "--loss", "cosine")
    call("src.train_bi", "--dataset", dataset, "--loss", "triplet")
    call("src.train_cross", "--dataset", dataset)
    call("src.train_llm", "--dataset", dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "analyze", "train", "evaluate", "report"])
    parser.add_argument("--dataset", choices=["all", "lcqmc", "bq_corpus"], default="all")
    args = parser.parse_args()
    datasets = ["lcqmc", "bq_corpus"] if args.dataset == "all" else [args.dataset]
    if args.stage in ("all", "analyze"):
        call("src.analyze", "--dataset", args.dataset)
    if args.stage in ("all", "train"):
        for dataset in datasets:
            train(dataset)
    if args.stage in ("all", "evaluate"):
        call("src.evaluate", "--dataset", args.dataset)
    if args.stage in ("all", "report"):
        call("src.report", "--dataset", args.dataset)


if __name__ == "__main__":
    main()
