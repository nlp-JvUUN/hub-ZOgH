import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 准备示例数据集（你可以替换成自己的csv） =====================
data = {
    'text': [
        '今天股票大涨 收益很好', '基金亏损 心态崩了', '汇率创新高 外贸受益',
        '外卖迟到 服务太差', '快递很快 好评', '电影超好看 推荐',
        '股价下跌 恐慌抛售', '银行理财稳定', '手机卡顿 质量差',
        '奶茶好喝 常回购', '外卖贵 不划算', '债券收益稳健',
        '游戏好玩 上瘾', '快递破损 投诉', '理财到期 回款顺利'
    ],
    'label': [0,0,0, 1,1,1, 0,0,1, 1,1,0, 1,1,0]  # 0=财经 1=生活
}
df = pd.DataFrame(data)
texts = df['text'].tolist()
labels = df['label'].tolist()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# ===================== 工具函数 =====================
def print_result(name, acc, t):
    print(f"【{name}】")
    print(f"准确率: {acc:.4f} | 耗时: {t:.2f}s\n")

# ==============================================================================
# ===================== 方法1：TF-IDF + 逻辑回归（传统机器学习） =====================
# ==============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

start = time.time()
tfidf = TfidfVectorizer(max_features=1000)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train_tfidf, y_train)
y_pred = lr.predict(x_test_tfidf)
print_result("TF-IDF + 逻辑回归", accuracy_score(y_test, y_pred), time.time()-start)

# ==============================================================================
# ======================== 方法2：FastText（轻量深度学习） ========================
# ==============================================================================
import fasttext

# 生成FastText需要的格式文件
with open("train.txt", "w", encoding="utf-8") as f:
    for t, l in zip(x_train, y_train):
        f.write(f"__label__{l} {t}\n")

start = time.time()
model = fasttext.train_supervised(
    input="train.txt",
    epoch=20,
    lr=0.1,
    wordNgrams=2,
    verbose=0
)

# 预测
preds = [model.predict(t)[0][0].replace("__label__", "") for t in x_test]
y_pred = [int(p) for p in preds]
print_result("FastText", accuracy_score(y_test, y_pred), time.time()-start)

# ==============================================================================
# ======================== 方法3：TextCNN（深度学习） ========================
# ==============================================================================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# 文本序列化
max_len = 30
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)
y_train_np = np.array(y_train)

# 构建TextCNN
start = time.time()
inputs = Input(shape=(max_len,))
embedding = Embedding(max_words, 64, input_length=max_len)(inputs)
conv1 = Conv1D(64, 3, activation='relu')(embedding)
pool1 = GlobalMaxPooling1D()(conv1)
drop = Dropout(0.3)(pool1)
outputs = Dense(1, activation='sigmoid')(drop)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练
model.fit(x_train_pad, y_train_np, epochs=10, batch_size=4, verbose=0)
y_pred_proba = model.predict(x_test_pad, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
print_result("TextCNN", accuracy_score(y_test, y_pred), time.time()-start)

# ==============================================================================
# ======================== 方法4：BERT 微调（预训练模型） ========================
# ==============================================================================
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

# 超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 32
batch_size = 2
epochs = 3
model_name = "bert-base-chinese"

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 加载BERT
start = time.time()
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 数据加载
train_dataset = TextDataset(x_train, y_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# 训练
model.train()
for _ in range(epochs):
    for batch in train_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# 预测
model.eval()
y_pred = []
with torch.no_grad():
    for t in x_test:
        enc = tokenizer(t, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
        out = model(**enc)
        pred = torch.argmax(out.logits, dim=1).item()
        y_pred.append(pred)

print_result("BERT", accuracy_score(y_test, y_pred), time.time()-start)