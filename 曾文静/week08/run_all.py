# -*- coding: utf-8 -*-
"""
run_all.py — 一键跑完一个数据集的三种方法（BiEncoder cosine / BiEncoder triplet / CrossEncoder）

用法：
  python run_all.py --dataset bq_corpus                 # BQ 全量，默认 3 epoch / 4 层
  python run_all.py --dataset lcqmc                     # LCQMC 全量（最久）
  python run_all.py --dataset lcqmc --quick 5000        # 快速验证：只用前 5000 条训练
  python run_all.py --dataset bq_corpus --epochs 5 --layers 12 --batch_size 16

流程（每个数据集独立归档，互不覆盖）：
  训练 3 个模型 → 评估 3 个 checkpoint → 产物存入 results/<dataset>/{checkpoints,logs,figures}
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PY = sys.executable
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"


def run(cmd, log_path):
    """执行子命令，输出追加写入 log 文件并实时打印尾部"""
    print(f"\n$ {' '.join(map(str, cmd))}")
    with open(log_path, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.rstrip()
            print(line[-160:])
            f.write(line + "\n")
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[失败] {' '.join(map(str, cmd))} 退出码 {proc.returncode}，见 {log_path}")


def train_one(args, method, tag):
    """训练一个模型，产物移动到 results/<dataset>/"""
    ds, out = args.dataset, args.out_dir
    for sub in ("checkpoints", "logs", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    logf = out / "logs" / f"{tag}.log"
    common = ["--data_dir", str(args.data_dir), "--bert_path", str(BERT_PATH),
              "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
              "--num_hidden_layers", str(args.layers)]
    if method == "cosine":
        cmd = [PY, str(ROOT / "src" / "train_biencoder.py"), "--loss", "cosine"] + common
        ckpt, jlog = "biencoder_cosine_best.pt", "biencoder_cosine_log.json"
    elif method == "triplet":
        cmd = [PY, str(ROOT / "src" / "train_biencoder.py"), "--loss", "triplet"] + common
        ckpt, jlog = "biencoder_triplet_best.pt", "biencoder_triplet_log.json"
    else:
        cmd = [PY, str(ROOT / "src" / "train_crossencoder.py")] + common
        ckpt, jlog = "crossencoder_best.pt", "crossencoder_log.json"

    run(cmd, logf)
    # 归档产物（脚本固定写 week08/outputs/，这里挪到独立目录）
    for src_name, dst_dir in [(ckpt, "checkpoints"), (jlog, "logs")]:
        s = ROOT / "outputs" / dst_dir / src_name
        if s.exists():
            shutil.move(str(s), out / dst_dir / src_name)
    return out / "logs" / jlog


def eval_one(args, model_type, ckpt_name, tag):
    out = args.out_dir
    logf = out / "logs" / f"{tag}.log"
    cmd = [PY, str(ROOT / "src" / "evaluate.py"),
           "--model_type", model_type,
           "--ckpt", str(out / "checkpoints" / ckpt_name),
           "--data_dir", str(args.data_dir),
           "--bert_path", str(BERT_PATH)]
    run(cmd, logf)
    fig = ROOT / "outputs" / "figures" / "biencoder_validation_sim_dist.png"
    if fig.exists():
        (out / "figures").mkdir(parents=True, exist_ok=True)
        shutil.move(str(fig), out / "figures" / f"{tag}_sim_dist.png")


def make_quick_dir(args):
    """quick 模式：从全量数据截取前 N 条训练、前 2000 条验证，加快验证流程"""
    src = ROOT / "data" / args.dataset
    tmp = ROOT / "tmp" / args.dataset
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_train = args.quick
    for split, limit in [("train", n_train), ("validation", min(n_train // 4, 2000)), ("test", 1000)]:
        rows = []
        with open(src / f"{split}.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                rows.append(line)
        with open(tmp / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(rows)
    print(f"[quick] 数据集缩小为 {n_train} 条训练 → {tmp}")
    return tmp


def main():
    parser = argparse.ArgumentParser(description="一个数据集 × 三种方法 全流程")
    parser.add_argument("--dataset", required=True, choices=["lcqmc", "bq_corpus"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4, help="BERT 层数（4 快速 / 12 完整）")
    parser.add_argument("--quick", type=int, default=0, help=">0 时只用前 N 条训练（快速验证）")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估步骤")
    args = parser.parse_args()

    assert BERT_PATH.exists(), f"预训练模型不存在: {BERT_PATH}，先运行 python download_bert.py"
    args.data_dir = make_quick_dir(args) if args.quick else ROOT / "data" / args.dataset
    args.out_dir = ROOT / "results" / (args.dataset if not args.quick else f"{args.dataset}_quick")

    t0 = time.time()
    print(f"===== 数据集 {args.dataset} | {args.epochs} epoch | {args.layers} 层 | batch {args.batch_size} =====")

    for method, tag in [("cosine", "train_cosine"), ("triplet", "train_triplet"), ("cross", "train_cross")]:
        train_one(args, method, tag)
        if not args.skip_eval:
            if method == "cross":
                eval_one(args, "crossencoder", "crossencoder_best.pt", "eval_cross")
            else:
                eval_one(args, "biencoder", f"biencoder_{method}_best.pt", f"eval_{method}")

    # 汇总打印
    print(f"\n===== {args.dataset} 训练+评估完成，总耗时 {(time.time()-t0)/60:.1f} 分钟 =====")
    for jlog in ["biencoder_cosine_log.json", "biencoder_triplet_log.json", "crossencoder_log.json"]:
        p = args.out_dir / "logs" / jlog
        if p.exists():
            recs = json.loads(p.read_text(encoding="utf-8"))
            best = max(recs, key=lambda r: r["val_f1"])
            print(f"  {jlog:28s} best_epoch={best['epoch']}  val_f1={best['val_f1']:.4f}  "
                  f"val_acc={best['val_acc']:.4f}  thr={best.get('threshold', 'argmax')}")
    if args.quick:
        shutil.rmtree(args.data_dir)
    print("产物目录:", args.out_dir)


if __name__ == "__main__":
    main()
