# -*- coding: utf-8 -*-
"""
week07/metrics.py
=================
序列标注评估工具:
  - entity-level P/R/F1 (seqeval, 按实体边界匹配)
  - 逐实体类型报告
  - 非法 BIO 序列统计 (CRF 效果量化的关键指标)
  - BIO 标签还原实体
"""
from seqeval.metrics import (classification_report as seq_report,
                             f1_score as seq_f1,
                             precision_score as seq_precision,
                             recall_score as seq_recall)


def entity_f1(true_tags, pred_tags):
    """整体 entity-level 指标"""
    return {
        "precision": round(seq_precision(true_tags, pred_tags), 4),
        "recall": round(seq_recall(true_tags, pred_tags), 4),
        "f1": round(seq_f1(true_tags, pred_tags), 4),
    }


def per_type_report(true_tags, pred_tags):
    """逐实体类型 P/R/F1 报告 (字符串表格)"""
    return seq_report(true_tags, pred_tags, digits=4)


def illegal_bio_stats(sequences):
    """
    统计非法 BIO 序列:
      - I 开头: 序列第一个标签就是 I-X
      - 非法转移: I-X 的前一个标签是 O 或类型不同的 B-Y/I-Y
    """
    n_start, n_trans = 0, 0
    for seq in sequences:
        if seq and seq[0].startswith("I-"):
            n_start += 1
        for prev, cur in zip(seq, seq[1:]):
            if not cur.startswith("I-"):
                continue
            if prev == "O" or (prev.startswith(("B-", "I-")) and prev[2:] != cur[2:]):
                n_trans += 1
    return {"非法I开头序列": n_start, "非法跨类型转移": n_trans,
            "总序列数": len(sequences)}


def recover_entities(tokens, tags):
    """
    从 BIO 标签还原实体列表:
      [(实体类型, 起始位置, 结束位置, 实体文本), ...]
    """
    entities, cur = [], None
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if cur:
                entities.append(cur)
            cur = [tag[2:], i, i, tok]
        elif tag.startswith("I-") and cur and cur[0] == tag[2:]:
            cur[2] = i
            cur[3] += tok
        else:
            if cur:
                entities.append(cur)
            cur = None
    if cur:
        entities.append(cur)
    return [(e[0], e[1], e[2], e[3]) for e in entities]
