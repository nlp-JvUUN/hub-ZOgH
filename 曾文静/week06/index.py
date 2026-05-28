# -*- coding: utf-8 -*-
"""
文本分类不同训练方法效果对比

对比四种有代表性的文本分类方法:
  1. TF-IDF + 逻辑回归   —— 传统统计机器学习
  2. 自实现 TextCNN      —— 卷积神经网络
  3. 自实现 BiLSTM       —— 双向循环神经网络
  4. DistilBERT 微调     —— 预训练语言模型(依赖 transformers, 未安装则自动跳过)

用法:
  python index.py            # 完整实验
  python index.py --fast     # 快速冒烟测试(减少轮次/词表, 验证整条流程)

输出文件:
  results.csv                       结果对比表
  text_classification_compare.png   四方法对比可视化
  best_model_confusion.png          最优模型的混淆矩阵
"""
import argparse
import csv
import re
import time
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= 超参数配置 =========================
DEFAULTS = dict(
    max_len=128, batch_size=64, max_vocab=10000, min_freq=2,
    epochs_cnn=12, epochs_lstm=12, epochs_bert=3,
    lr_cnn=1e-3, lr_lstm=1e-3, lr_bert=2e-5,
)
FAST = dict(max_len=64, batch_size=32, max_vocab=3000, min_freq=1,
            epochs_cnn=2, epochs_lstm=2, epochs_bert=1)

CATEGORIES = ["rec.autos", "rec.motorcycles", "sci.space", "talk.politics.misc"]


# ========================= 文本预处理 =========================
def clean_text(text):
    """统一小写, 只保留字母和数字, 用空格连接"""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_vocab(texts, max_vocab, min_freq):
    """按词频构建词表, 过滤低频词, 保留<PAD>/<UNK>"""
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, cnt in counter.most_common():
        if len(vocab) >= max_vocab or cnt < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts, vocab, max_len):
    """文本 -> (token id 矩阵, 掩码矩阵)"""
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    mask = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, 1) for w in t.split()[:max_len]]
        ids[i, :len(toks)] = toks
        mask[i, :len(toks)] = 1
    return ids, mask


class ClfDataset(Dataset):
    """文本分类数据集: (ids, mask, label)"""

    def __init__(self, ids, mask, labels):
        self.ids, self.mask, self.labels = ids, mask, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i]),
                torch.from_numpy(self.mask[i]),
                torch.tensor(self.labels[i], dtype=torch.long))


# ========================= 自实现模型 =========================
class TextCNN(nn.Module):
    """TextCNN: 多尺寸卷积核 + 自适应最大池化"""

    def __init__(self, vocab_size, embed_dim=100, n_filters=128,
                 filter_sizes=(2, 3, 4), n_classes=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_dim, n_filters, k),
                          nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, x, mask=None):
        emb = self.embedding(x).transpose(1, 2)          # (B, D, L)
        feats = [conv(emb).squeeze(-1) for conv in self.convs]  # 每个尺寸一个特征
        out = self.dropout(torch.cat(feats, dim=1))
        return self.classifier(out)


class BiLSTM(nn.Module):
    """双向LSTM: 取最后一层前向/后向隐状态拼接分类"""

    def __init__(self, vocab_size, embed_dim=100, hidden=128, n_layers=2,
                 n_classes=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, mask=None):
        lengths = mask.sum(dim=1).clamp(min=1).cpu()     # pack需要CPU上的长度
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)             # 最后的前向+后向隐状态
        return self.classifier(self.dropout(h))


# ========================= 训练与评估 =========================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader):
    """返回 (准确率, F1_macro, 推理耗时)"""
    model.eval()
    preds, labels = [], []
    t0 = time.time()
    for ids, mask, label in loader:
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        preds.extend(model(ids, mask).argmax(1).cpu().tolist())
        labels.extend(label.tolist())
    infer_time = time.time() - t0
    return (accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro"), infer_time)


@torch.no_grad()
def predict(model, loader):
    """返回测试集全部预测标签(用于混淆矩阵)"""
    model.eval()
    preds = []
    for ids, mask, _ in loader:
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        preds.extend(model(ids, mask).argmax(1).cpu().tolist())
    return np.array(preds)


def train_model(model, train_loader, test_loader, epochs, lr, name=""):
    """通用训练流程, 记录每轮损失/测试准确率历史"""
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "acc": []}
    t0 = time.time()

    print(f"\n>>> 训练 {name} | 参数量 {count_params(model):,} | 设备 {DEVICE}")
    print("-" * 64)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ids.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += ids.size(0)
        test_acc, test_f1, _ = evaluate(model, test_loader)
        history["loss"].append(total_loss / total)
        history["acc"].append(test_acc)
        print(f"  epoch {epoch:2d}/{epochs} | loss {total_loss/total:.4f} "
              f"| train_acc {correct/total:.4f} | test_acc {test_acc:.4f} | f1 {test_f1:.4f}")

    train_time = time.time() - t0
    test_acc, test_f1, infer_time = evaluate(model, test_loader)
    return {
        "name": name, "params": count_params(model),
        "train_time": round(train_time, 2), "infer_time": round(infer_time, 4),
        "accuracy": round(test_acc, 4), "f1": round(test_f1, 4),
        "epochs": epochs, "history": history, "model": model,
    }


# ========================= 方法一: TF-IDF + 逻辑回归 =========================
def run_tfidf_lr(train_texts, train_labels, test_texts, test_labels):
    print("\n>>> 方法一: TF-IDF + 逻辑回归")
    print("-" * 64)
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, train_labels)
    train_time = time.time() - t0

    t1 = time.time()
    preds = clf.predict(X_test)
    infer_time = time.time() - t1

    n_params = X_train.shape[1] * clf.coef_.shape[0] + clf.intercept_.size
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    print(f"  特征维度 {X_train.shape[1]:,} | 训练时间 {train_time:.1f}s | "
          f"测试准确率 {acc:.4f} | F1 {f1:.4f}")
    return {"name": "TF-IDF+逻辑回归", "params": n_params,
            "train_time": round(train_time, 2), "infer_time": round(infer_time, 4),
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "epochs": 1, "history": None, "model": None, "preds": preds}


# ========================= 方法四: DistilBERT 微调 =========================
def run_distilbert(train_texts, train_labels, test_texts, test_labels, cfg):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "distilbert-base-uncased"
    print(f"\n>>> 方法四: DistilBERT 微调 ({model_name})")
    print("-" * 64)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(train_texts, padding=True, truncation=True,
                    max_length=cfg["max_len"], return_tensors="pt")
    enc_test = tokenizer(test_texts, padding=True, truncation=True,
                         max_length=cfg["max_len"], return_tensors="pt")

    train_set = ClfDataset(enc["input_ids"].numpy(), enc["attention_mask"].numpy(),
                           np.asarray(train_labels))
    test_set = ClfDataset(enc_test["input_ids"].numpy(), enc_test["attention_mask"].numpy(),
                          np.asarray(test_labels))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(set(train_labels)))
    return train_model(model, train_loader, test_loader,
                       epochs=cfg["epochs_bert"], lr=cfg["lr_bert"],
                       name="DistilBERT(微调)")


# ========================= 结果输出 =========================
def save_csv(results):
    fields = ["name", "params", "train_time", "infer_time", "accuracy", "f1", "epochs"]
    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\n[保存] 结果表 -> results.csv")


def print_table(results):
    print(f"\n{'模型':<18}{'参数量':>12}{'训练时间(s)':>12}{'推理时间(s)':>12}"
          f"{'准确率':>10}{'F1':>10}{'轮数':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<18}{r['params']:>12,}{r['train_time']:>12.1f}"
              f"{r['infer_time']:>12.4f}{r['accuracy']:>10.4f}{r['f1']:>10.4f}"
              f"{r['epochs']:>6}")


def make_plots(results, target_names):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    times = [r["train_time"] for r in results]
    best = max(results, key=lambda r: r["accuracy"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = np.arange(len(names))
    w = 0.38

    # (a) 准确率与F1
    ax = axes[0, 0]
    ax.bar(x - w / 2, accs, w, label="准确率", color="#4C72B0")
    ax.bar(x + w / 2, f1s, w, label="F1(macro)", color="#DD8452")
    for xi, a, f in zip(x, accs, f1s):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("分数")
    ax.set_title("(a) 准确率与F1对比")
    ax.legend()

    # (b) 训练损失曲线
    ax = axes[0, 1]
    for r in results:
        if r["history"]:
            ax.plot(range(1, r["epochs"] + 1), r["history"]["loss"],
                    marker="o", markersize=3, label=r["name"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("训练损失")
    ax.set_title("(b) 训练损失曲线")
    ax.legend()

    # (c) 训练时间
    ax = axes[1, 0]
    bars = ax.bar(names, times, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t, f"{t:.0f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("训练时间(秒)")
    ax.set_title("(c) 训练时间对比")

    # (d) 总结
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["实验总结",
             f"最优方法: {best['name']}",
             f"最高准确率: {best['accuracy']:.1%}",
             f"最高F1: {max(r['f1'] for r in results):.1%}",
             f"对比方法数: {len(results)}",
             f"运行设备: {DEVICE}"]
    lines += [f"{r['name']}: {r['params']/1e6:.1f}M 参数" for r in results]
    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0"))

    fig.tight_layout()
    fig.savefig("text_classification_compare.png", dpi=150, bbox_inches="tight")
    print("[保存] 对比图 -> text_classification_compare.png")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo"):
        plt.show()
    plt.close(fig)


def plot_confusion(cm, target_names, fname="best_model_confusion.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticks(range(len(target_names)))
    ax.set_yticklabels(target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("最优模型混淆矩阵")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[保存] 混淆矩阵 -> {fname}")
    plt.close(fig)


# ========================= 主流程 =========================
def main():
    parser = argparse.ArgumentParser(description="文本分类不同训练方法效果对比")
    parser.add_argument("--fast", action="store_true", help="快速冒烟测试模式")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **FAST} if args.fast else dict(DEFAULTS)

    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[INFO] 设备: {DEVICE} | 模式: {'快速' if args.fast else '完整'}")

    # ---------- 1. 加载数据 ----------
    print("\n" + "=" * 56)
    print("1. 加载 20 Newsgroups 数据集")
    print("=" * 56)
    news_train = fetch_20newsgroups(subset="train", categories=CATEGORIES,
                                    remove=("headers", "footers", "quotes"))
    news_test = fetch_20newsgroups(subset="test", categories=CATEGORIES,
                                   remove=("headers", "footers", "quotes"))
    train_texts = [clean_text(t) for t in news_train.data]
    test_texts = [clean_text(t) for t in news_test.data]
    train_labels = list(news_train.target)
    test_labels = list(news_test.target)
    print(f"训练集: {len(train_texts)} 条 | 测试集: {len(test_texts)} 条")
    print(f"类别: {news_train.target_names}")
    print(f"平均长度(词): 训练 {np.mean([len(t.split()) for t in train_texts]):.0f} | "
          f"测试 {np.mean([len(t.split()) for t in test_texts]):.0f}")

    # ---------- 2. 构建词表与数据加载器 ----------
    print("\n" + "=" * 56)
    print("2. 构建词表与数据加载器")
    print("=" * 56)
    vocab = build_vocab(train_texts, cfg["max_vocab"], cfg["min_freq"])
    print(f"词表大小: {len(vocab)} (max_vocab={cfg['max_vocab']}, min_freq={cfg['min_freq']})")

    train_ids, train_mask = encode_texts(train_texts, vocab, cfg["max_len"])
    test_ids, test_mask = encode_texts(test_texts, vocab, cfg["max_len"])
    train_loader = DataLoader(ClfDataset(train_ids, train_mask, train_labels),
                              batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(ClfDataset(test_ids, test_mask, test_labels),
                             batch_size=cfg["batch_size"], shuffle=False)

    # ---------- 3. 依次训练四种方法 ----------
    results = []

    try:
        results.append(run_tfidf_lr(train_texts, train_labels, test_texts, test_labels))
    except Exception as e:
        print(f"[警告] TF-IDF+逻辑回归失败: {type(e).__name__}: {e}")

    try:
        results.append(train_model(TextCNN(len(vocab), n_classes=len(CATEGORIES)),
                                   train_loader, test_loader,
                                   epochs=cfg["epochs_cnn"], lr=cfg["lr_cnn"],
                                   name="TextCNN(自实现)"))
    except Exception as e:
        print(f"[警告] TextCNN失败: {type(e).__name__}: {e}")

    try:
        results.append(train_model(BiLSTM(len(vocab), n_classes=len(CATEGORIES)),
                                   train_loader, test_loader,
                                   epochs=cfg["epochs_lstm"], lr=cfg["lr_lstm"],
                                   name="BiLSTM(自实现)"))
    except Exception as e:
        print(f"[警告] BiLSTM失败: {type(e).__name__}: {e}")

    try:
        results.append(run_distilbert(train_texts, train_labels,
                                      test_texts, test_labels, cfg))
    except Exception as e:
        print(f"[警告] DistilBERT微调失败: {type(e).__name__}: {e}，跳过该方法")

    if not results:
        print("[错误] 所有方法均失败, 无法对比")
        return

    # ---------- 4. 结果对比与可视化 ----------
    print("\n" + "=" * 56)
    print("4. 结果对比")
    print("=" * 56)
    print_table(results)
    save_csv(results)
    make_plots(results, news_train.target_names)

    best = max(results, key=lambda r: r["accuracy"])
    preds = best["preds"] if best["preds"] is not None \
        else predict(best["model"], test_loader)
    cm = confusion_matrix(test_labels, preds)
    plot_confusion(cm, news_train.target_names)

    print(f"\n最优方法 [{best['name']}] 的类别级指标:")
    print(classification_report(test_labels, preds,
                                target_names=news_train.target_names, digits=4))
    print("\n实验完成!")


if __name__ == "__main__":
    main()
