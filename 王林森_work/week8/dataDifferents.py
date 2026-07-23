import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel

# ====================== 工具函数：评估指标 ======================
def get_metrics(y_true, y_pred):
    return {
        "Acc": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4)
    }

# ====================== 加载数据集 ======================
def load_data(file_path):
    df = pd.read_csv(file_path)
    s1 = df["text1"].tolist()
    s2 = df["text2"].tolist()
    label = df["label"].tolist()
    return s1, s2, label

# ====================== 方法1 TF-IDF + 余弦相似度 ======================
def method_tfidf(s1_list, s2_list, labels, threshold=0.5):
    all_text = s1_list + s2_list
    vec = TfidfVectorizer()
    mat = vec.fit_transform(all_text)
    mat1 = mat[:len(s1_list)]
    mat2 = mat[len(s1_list):]
    preds = []
    for i in range(len(s1_list)):
        sim = (mat1[i] @ mat2[i].T).toarray()[0][0]
        preds.append(1 if sim >= threshold else 0)
    return get_metrics(labels, preds)

# ====================== 方法2 BM25 ======================
def method_bm25(s1_list, s2_list, labels, threshold=1.0):
    token_s1 = [text.split() for text in s1_list]
    bm25 = BM25Okapi(token_s1)
    preds = []
    for text in s2_list:
        score = bm25.get_scores(text.split())[0]
        preds.append(1 if score >= threshold else 0)
    return get_metrics(labels, preds)

# ====================== 方法3 BERT句向量余弦匹配 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name).to(device)
bert.eval()

def get_cls_emb(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=64
    ).to(device)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:, 0, :]

def method_bert_cosine(s1_list, s2_list, labels, threshold=0.7):
    preds = []
    for s1, s2 in zip(s1_list, s2_list):
        e1 = get_cls_emb(s1)
        e2 = get_cls_emb(s2)
        sim = torch.cosine_similarity(e1, e2).item()
        preds.append(1 if sim >= threshold else 0)
    return get_metrics(labels, preds)

# ====================== 批量跑两个数据集 ======================
def run_all_methods(ds1_path, ds2_path):
    datasets = {
        "数据集1": load_data(ds1_path),
        "数据集2": load_data(ds2_path)
    }
    methods = {
        "TF-IDF余弦": method_tfidf,
        "BM25": method_bm25,
        "BERT-CLS余弦": method_bert_cosine
    }
    total_result = {}
    for ds_name, (s1, s2, y_true) in datasets.items():
        print(f"\n========== 正在测试 {ds_name} ==========")
        res = {}
        for meth_name, func in methods.items():
            print(f"运行 {meth_name} ...")
            res[meth_name] = func(s1, s2, y_true)
        total_result[ds_name] = res
    return total_result

# ====================== 打印对比表格 + 保存Excel ======================
def show_and_save(results, save_path="文本匹配实验结果.xlsx"):
    rows = []
    for ds_name, meth_dict in results.items():
        for meth, metric in meth_dict.items():
            row = [ds_name, meth, metric["Acc"], metric["Precision"], metric["Recall"], metric["F1"]]
            rows.append(row)
    df_res = pd.DataFrame(rows, columns=["数据集", "方法", "Acc", "Precision", "Recall", "F1"])
    print("\n==================== 实验总结果 ====================")
    print(df_res.to_string(index=False))
    df_res.to_excel(save_path, index=False)
    print(f"\n结果已保存至 {save_path}")

# ====================== 主入口 ======================
if __name__ == "__main__":
    path1 = "data1.csv"
    path2 = "data2.csv"
    result = run_all_methods(path1, path2)
    show_and_save(result)
