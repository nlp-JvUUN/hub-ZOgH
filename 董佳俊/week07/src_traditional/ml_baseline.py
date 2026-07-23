"""
传统机器学习 Baseline：TF-IDF + 经典分类器

目的：为深度学习方法提供"地板线"参考——如果 BERT 全量微调只比 TF-IDF+LR
高几个点，那说明这个任务太简单了，不值得上大模型。

对比分类器：
  - Logistic Regression（线性基准）
  - Linear SVM（最大间隔）
  - XGBoost（树模型天花板，需 pip install xgboost）

输出：每条方法的 accuracy / precision / recall / f1（macro & weighted）
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"


def load_data():
    """延迟导入，避免循环依赖。"""
    import sys
    sys.path.insert(0, str(ROOT))
    from data_loader import load_dataset, get_label_names

    train = load_dataset("train")
    val = load_dataset("validation")
    test = load_dataset("test")
    label_names = get_label_names()
    return train, val, test, label_names


def _as_xy(records: list[dict]) -> tuple[list[str], list[int]]:
    X = [r["text"] for r in records]
    y = [r["label"] for r in records]
    return X, y


def train_and_eval(
        name: str,
        clf,
        X_train: list[str], y_train: list[int],
        X_test: list[str], y_test: list[int],
        label_names: dict[int, str],
) -> dict:
    """训练 + 评估，返回指标字典。"""
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = clf.predict(X_test)
    target_names = [label_names[i] for i in sorted(label_names)]

    acc = accuracy_score(y_test, y_pred)
    p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    result = {
        "method": name,
        "accuracy": round(acc, 6),
        "precision_macro": round(p_macro, 6),
        "recall_macro": round(r_macro, 6),
        "f1_macro": round(f1_macro, 6),
        "f1_weighted": round(f1_weighted, 6),
        "train_time_s": round(elapsed, 2),
    }

    print(f"\n{'─' * 55}")
    print(f"  [{name}]")
    print(f"    训练耗时: {elapsed:.2f}s")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1-macro : {f1_macro:.4f}")
    print(f"    F1-wted  : {f1_weighted:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0))

    return result


def run_tfidf_lr(X_train, y_train, X_test, y_test, label_names) -> dict:
    """TF-IDF + Logistic Regression（线性基础模型）。"""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")),
    ])
    return train_and_eval("TF-IDF + LR", pipe, X_train, y_train, X_test, y_test, label_names)


def run_tfidf_svm(X_train, y_train, X_test, y_test, label_names) -> dict:
    """TF-IDF + Linear SVM（最大间隔分类器）。"""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(max_iter=2000, C=1.0, dual=False, random_state=42)),
    ])
    return train_and_eval("TF-IDF + SVM", pipe, X_train, y_train, X_test, y_test, label_names)


def run_tfidf_xgb(X_train, y_train, X_test, y_test, label_names) -> dict | None:
    """TF-IDF + XGBoost（树模型天花板），如果未安装则跳过。"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [跳过] xgboost 未安装（pip install xgboost）")
        return None

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )),
    ])
    return train_and_eval("TF-IDF + XGBoost", pipe, X_train, y_train, X_test, y_test, label_names)


def main():
    args = parse_args()
    print("=" * 60)
    print("  传统机器学习 Baseline —— 文本分类")
    print("=" * 60)

    train, val, test, label_names = load_data()

    # 合并 train + val 作为训练集（传统 ML 不需要 validation set 做 early stopping）
    combined = train + val
    X_train, y_train = _as_xy(combined)
    X_test, y_test = _as_xy(test)

    print(f"\n训练数据: {len(combined)} 条 (train + validation)")
    print(f"测试数据: {len(test)} 条")

    results = []

    # 逐个跑分类器
    for runner in [run_tfidf_lr, run_tfidf_svm, run_tfidf_xgb]:
        res = runner(X_train, y_train, X_test, y_test, label_names)
        if res:
            results.append(res)

    # ── 保存 ──
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "ml_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ML Baseline 结果已保存 → {out_path}")
    print(f"  下一步：python src_bert/bert_trainer.py --method full")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(description="传统 ML 文本分类 baseline")
    return parser.parse_args()


if __name__ == "__main__":
    main()
