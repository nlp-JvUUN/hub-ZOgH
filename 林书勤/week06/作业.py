
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {DEVICE}")

# ========================= 超参数配置 =========================
BATCH_SIZE = 32
MAX_LEN = 128
NUM_EPOCHS_TRANSFORMER = 20
NUM_EPOCHS_BERT = 5
LR_TRANSFORMER = 1e-3
LR_BERT = 2e-5
NUM_CLASSES = 4

# ========================= 数据加载 =========================
print("\n" + "="*50)
print("1. 加载20 Newsgroups数据集")
print("="*50)

categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
news_train = fetch_20newsgroups(subset='train', categories=categories)
news_test = fetch_20newsgroups(subset='test', categories=categories)

print(f"训练集: {len(news_train.data)} 条")
print(f"测试集: {len(news_test.data)} 条")
print(f"类别: {news_train.target_names}")

# ========================= 构建词汇表 =========================
from collections import Counter

def build_vocab(texts, max_vocab=10000):
    """构建词汇表"""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(news_train.data, max_vocab=10000)
print(f"词汇表大小: {len(vocab)}")

# ========================= 数据预处理 =========================
def tokenize(text, vocab, max_len=MAX_LEN):
    """将文本转换为token ID序列"""
    tokens = [vocab.get(w, 1) for w in text.lower().split()[:max_len]]
    mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens), torch.tensor(mask)

class TextDataset(Dataset):
    """文本分类数据集类"""
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        ids, mask = tokenize(self.texts[idx], self.vocab, self.max_len)
        return ids, mask, torch.tensor(self.labels[idx])

# 创建数据集和数据加载器
train_dataset = TextDataset(news_train.data, news_train.target, vocab)
test_dataset = TextDataset(news_test.data, news_test.target, vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================= 自实现Transformer模型 =========================
print("\n" + "="*50)
print("2. 自实现Transformer模型")
print("="*50)

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class TransformerClassifier(nn.Module):
    """Transformer分类器"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, 
                 n_layers=3, n_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, mask)
        x = x[:, 0, :]  # 使用[CLS]位置
        return self.classifier(x)

# ========================= 训练和评估函数 =========================
def count_params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer, criterion):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ids.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += ids.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    """评估模型性能"""
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        logits = model(ids, mask)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    infer_time = time.time() - t0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, infer_time

def train_model(model, train_loader, test_loader, epochs, lr, name=""):
    """训练模型主函数"""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t0 = time.time()
    
    print(f"\n训练: {name} | 参数量: {count_params(model):,}")
    print("-"*50)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc, test_f1, _ = evaluate(model, test_loader)
        best_acc = max(best_acc, test_acc)
        
        if epoch % (epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Loss={train_loss:.4f} | "
                  f"Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f} | Test F1={test_f1:.4f}")
    
    train_time = time.time() - t0
    test_acc, test_f1, infer_time = evaluate(model, test_loader)
    
    return {
        'name': name,
        'params': count_params(model),
        'train_time': round(train_time, 2),
        'infer_time': round(infer_time, 4),
        'accuracy': round(test_acc, 4),
        'f1': round(test_f1, 4)
    }

# ========================= 训练Transformer模型 =========================
print("\n" + "="*50)
print("3. 训练自实现Transformer")
print("="*50)

transformer_model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=3,
    n_classes=NUM_CLASSES
)

result_transformer = train_model(
    transformer_model, train_loader, test_loader,
    epochs=NUM_EPOCHS_TRANSFORMER, lr=LR_TRANSFORMER,
    name="自实现Transformer"
)

# ========================= 微调BERT模型 =========================
print("\n" + "="*50)
print("4. 微调预训练BERT")
print("="*50)

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    
    # 加载BERT tokenizer和模型
    bert_name = 'bert-base-uncased'
    tokenizer_bert = BertTokenizer.from_pretrained(bert_name)
    
    def tokenize_bert(texts, max_len=MAX_LEN):
        """BERT tokenize函数"""
        enc = tokenizer_bert(
            list(texts), padding=True, truncation=True,
            max_length=max_len, return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']
    
    # 准备BERT数据
    train_ids, train_mask = tokenize_bert(news_train.data)
    test_ids, test_mask = tokenize_bert(news_test.data)
    
    train_labels = torch.tensor(news_train.target)
    test_labels = torch.tensor(news_test.target)
    
    class BERTDataset(Dataset):
        def __init__(self, ids, mask, labels):
            self.ids = ids
            self.mask = mask
            self.labels = labels
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            return self.ids[idx], self.mask[idx], self.labels[idx]
    
    bert_train_set = BERTDataset(train_ids, train_mask, train_labels)
    bert_test_set = BERTDataset(test_ids, test_mask, test_labels)
    bert_train_loader = DataLoader(bert_train_set, batch_size=BATCH_SIZE, shuffle=True)
    bert_test_loader = DataLoader(bert_test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 加载BERT模型
    bert_model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=NUM_CLASSES)
    
    # 微调BERT
    result_bert = train_model(
        bert_model, bert_train_loader, bert_test_loader,
        epochs=NUM_EPOCHS_BERT, lr=LR_BERT,
        name="BERT(微调)"
    )

except ImportError:
    print("未安装transformers库，跳过BERT部分")
    print("安装命令: pip install transformers")
    result_bert = None

# ========================= 结果对比 =========================
print("\n" + "="*50)
print("5. 结果对比")
print("="*50)

# 汇总结果
all_results = [result_transformer]
if result_bert:
    all_results.append(result_bert)

# 打印结果表格
print(f"\n{'模型':<20} {'参数量':>10} {'训练时间(s)':>12} {'推理时间(s)':>12} {'准确率':>10} {'F1分数':>10}")
print("-"*80)
for r in all_results:
    print(f"{r['name']:<20} {r['params']:>10,} {r['train_time']:>12.1f} "
          f"{r['infer_time']:>12.4f} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

# ========================= 可视化结果 =========================
if len(all_results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = [r['name'] for r in all_results]
    accs = [r['accuracy'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    times = [r['train_time'] for r in all_results]
    params = [r['params'] for r in all_results]
    
    # 准确率和F1分数对比
    x = np.arange(len(names))
    width = 0.35
    axes[0,0].bar(x - width/2, accs, width, label='准确率', color='skyblue')
    axes[0,0].bar(x + width/2, f1s, width, label='F1分数', color='lightcoral')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(names)
    axes[0,0].set_ylabel('分数')
    axes[0,0].set_title('准确率和F1分数对比')
    axes[0,0].legend()
    axes[0,0].set_ylim(0, 1)
    
    # 训练时间对比
    axes[0,1].bar(names, times, color=['skyblue', 'lightcoral'])
    axes[0,1].set_ylabel('训练时间(秒)')
    axes[0,1].set_title('训练时间对比')
    
    # 参数量对比
    axes[1,0].bar(names, [p/1e6 for p in params], color=['skyblue', 'lightcoral'])
    axes[1,0].set_ylabel('参数量(百万)')
    axes[1,0].set_title('参数量对比')
    
    # 综合对比雷达图
    axes[1,1].axis('off')
    axes[1,1].text(0.5, 0.5, '实验总结:\n' +
                  f'1. BERT准确率: {accs[1]:.1%}\n' +
                  f'2. Transformer准确率: {accs[0]:.1%}\n' +
                  f'3. BERT参数量: {params[1]/1e6:.1f}M\n' +
                  f'4. Transformer参数量: {params[0]/1e6:.1f}M',
                  transform=axes[1,1].transAxes,
                  verticalalignment='center',
                  horizontalalignment='center',
                  fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('transformer_vs_bert.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n对比图已保存为 transformer_vs_bert.png")

print("\n实验完成!")
