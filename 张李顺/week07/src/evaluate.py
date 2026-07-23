import argparse
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

from .bm25 import BM25
from .common import CFG, clean_rows, load_rows, output_dir
from .llm import attach_adapter, load_base, scores as llm_scores
from .models import cross_scores, encode, load_bi, load_cross, tokenizer


def release(*models):
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def best_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    values = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thresholds[int(np.nanargmax(values[:-1]))])


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }, pred


def bi_pair(path, val, test, tok):
    model = load_bi(path)
    va = encode(model, tok, [x["sentence1"] for x in val])
    vb = encode(model, tok, [x["sentence2"] for x in val])
    ta = encode(model, tok, [x["sentence1"] for x in test])
    tb = encode(model, tok, [x["sentence2"] for x in test])
    release(model)
    return np.sum(va * vb, axis=1), np.sum(ta * tb, axis=1)


def cross_pair(path, val, test, tok):
    model = load_cross(path)
    val_score = cross_scores(model, tok, val)
    test_score = cross_scores(model, tok, test)
    release(model)
    return val_score, test_score


def normalize(reference, values):
    return (values - reference.mean()) / max(reference.std(), 1e-8)


def write_badcases(dataset, rows, predictions, scores, thresholds):
    labels = np.array([x["label"] for x in rows])
    records = []
    for method, pred in predictions.items():
        wrong = np.where(pred != labels)[0]
        for i in wrong:
            row = rows[i]
            records.append({
                "method": method,
                "error": "false_positive" if pred[i] else "false_negative",
                "label": row["label"],
                "prediction": int(pred[i]),
                "score": float(scores[method][i]),
                "threshold": float(thresholds[method]),
                "wrong_confidence": float(abs(scores[method][i] - thresholds[method])),
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"]
            })
    frame = pd.DataFrame(records).sort_values(["method", "wrong_confidence"], ascending=[True, False])
    target = output_dir(dataset)
    frame.to_csv(target / "badcases.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["method", "error"]).size().unstack(fill_value=0).reset_index()
    for name in ("false_positive", "false_negative"):
        if name not in summary:
            summary[name] = 0
    summary["errors"] = summary["false_positive"] + summary["false_negative"]
    summary["error_rate"] = summary["errors"] / len(rows)
    summary.to_csv(target / "badcase_summary.csv", index=False, encoding="utf-8-sig")
    lines = [f"# {dataset} Badcase", "", f"测试样本：{len(rows)}。CSV 保存全部预测错误，不做自动原因分类。", ""]
    for method in predictions:
        row = summary[summary.method == method].iloc[0]
        lines.extend([f"## {method}", "", f"- FP：{row.false_positive}", f"- FN：{row.false_negative}", f"- 总错误：{row.errors}", f"- 错误率：{row.error_rate:.4f}", ""])
    (target / "badcase_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(dataset):
    val = clean_rows(load_rows(dataset, "validation"))
    test = clean_rows(load_rows(dataset, "test"))
    yv = np.array([x["label"] for x in val])
    yt = np.array([x["label"] for x in test])
    tok = tokenizer()
    val_scores, test_scores, thresholds, predictions = {}, {}, {}, {}
    for name in ("bi_cosine", "bi_triplet"):
        val_scores[name], test_scores[name] = bi_pair(output_dir(dataset, name), val, test, tok)
    val_scores["CrossEncoder"], test_scores["CrossEncoder"] = cross_pair(output_dir(dataset, "cross"), val, test, tok)
    best_bi = max(("bi_cosine", "bi_triplet"), key=lambda x: f1_score(yv, val_scores[x] >= best_threshold(yv, val_scores[x])))
    docs = [x["sentence2"] for x in clean_rows(load_rows(dataset, "train"))[:CFG["eval"]["bm25_train_docs"]]]
    bm = BM25(docs)
    bm_val = np.array([bm.pair_score(x["sentence1"], x["sentence2"]) for x in val])
    bm_test = np.array([bm.pair_score(x["sentence1"], x["sentence2"]) for x in test])
    val_scores["BiEncoder + BM25"] = .7 * normalize(val_scores[best_bi], val_scores[best_bi]) + .3 * normalize(bm_val, bm_val)
    test_scores["BiEncoder + BM25"] = .7 * normalize(val_scores[best_bi], test_scores[best_bi]) + .3 * normalize(bm_val, bm_test)
    base, llm_tok = load_base()
    val_scores["LLM Zero-shot"] = llm_scores(base, llm_tok, val, 64)
    test_scores["LLM Zero-shot"] = llm_scores(base, llm_tok, test, 64)
    lora = attach_adapter(base, output_dir(dataset, "llm_lora"))
    val_scores["LLM SFT LoRA"] = llm_scores(lora, llm_tok, val, 64)
    test_scores["LLM SFT LoRA"] = llm_scores(lora, llm_tok, test, 64)
    release(lora)
    records = []
    for method in test_scores:
        threshold = best_threshold(yv, val_scores[method])
        result, pred = metrics(yt, test_scores[method], threshold)
        records.append({"dataset": dataset, "method": method, "validation_rows": len(val), "test_rows": len(test), **result})
        thresholds[method] = threshold
        predictions[method] = pred
    target = output_dir(dataset)
    frame = pd.DataFrame(records)
    frame.to_csv(target / "pair_metrics.csv", index=False, encoding="utf-8-sig")
    pred_frame = pd.DataFrame(test)
    for method in test_scores:
        pred_frame[f"{method}_score"] = test_scores[method]
        pred_frame[f"{method}_pred"] = predictions[method]
    pred_frame.to_csv(target / "pair_predictions.csv", index=False, encoding="utf-8-sig")
    write_badcases(dataset, test, predictions, test_scores, thresholds)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *CFG["datasets"]], default="all")
    args = parser.parse_args()
    datasets = list(CFG["datasets"]) if args.dataset == "all" else [args.dataset]
    frames = [evaluate(dataset) for dataset in datasets]
    pd.concat(frames).to_csv(output_dir("summary") / "pair_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
