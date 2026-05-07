"""
train_chinese_multicls_rnn_lstm.py
中文句子场景多分类 —— RNN vs LSTM 对比实验

任务：根据句子内容判断所属场景（5分类）
  0-购物消费  1-学习教育  2-旅行出行  3-工作职场  4-娱乐休闲

模型对比：
  - RNN:  Embedding → RNN → MaxPool → FC
  - LSTM: Embedding → LSTM → MaxPool → FC

数据构造：多模板 + 多关键词组合，更贴近自然语言
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 1224
N_SAMPLES   = 5000          # 总样本数
MAXLEN      = 40            # 最大序列长度
EMBED_DIM   = 128           # 词向量维度
HIDDEN_DIM  = 128           # 隐藏层维度
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 15
TRAIN_RATIO = 0.8

N_CLASSES   = 5             # 分类数

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据构造（更自然的方式）────────────────────────────

# 每个场景的典型句式模板
SCENE_TEMPLATES = {
    0: [  # 购物消费
        "今天在{}买了一件{}，价格还挺{}",
        "这个{}的质量{}，下次还会回购",
        "{}正在搞活动，{}打{}折",
        "刚下单了{}，{}物流{}发货",
        "对比了几家{}，最后选了这款{}",
        "{}的{}款式很{}，很适合我",
        "网购{}真的{}，省了不少钱",
        "实体店试了{}，感觉{}",
    ],
    1: [  # 学习教育
        "今天{}课上讲了{}，感觉{}",
        "这道{}题目{}，我花了{}才解出来",
        "{}考试快到了，正在复习{}",
        "老师布置了{}作业，{}完成",
        "图书馆借了本{}，内容{}",
        "学习{}真的{}，需要多练习",
        "这个{}知识点{}，考试必考",
        "{}成绩出来了，{}考得{}",
    ],
    2: [  # 旅行出行
        "去{}旅游了{}天，景色{}",
        "{}的{}真的{}，值得一看",
        "订了{}的酒店，{}很{}",
        "坐{}去{}，路上花了{}",
        "这个{}景点{}，拍照很出片",
        "{}旅游攻略说{}，实际{}",
        "假期去了{}，体验了当地{}",
        "{}的{}很有特色，{}",
    ],
    3: [  # 工作职场
        "今天{}会议讨论了{}，结论是{}",
        "{}项目{}了，领导{}",
        "同事{}帮我{}，很感激",
        "加班到{}，终于把{}做完了",
        "这个季度{}目标{}，需要努力",
        "{}部门新来了个{}，{}",
        "工作{}年，学到很多{}",
        "公司{}政策{}，大家反应{}",
    ],
    4: [  # 娱乐休闲
        "周末{}看了部{}电影，剧情{}",
        "和朋友{}玩{}，玩得很开心",
        "{}新出了一款{}游戏，{}",
        "昨晚{}追剧到{}，今天好困",
        "这首歌{}，{}很好听",
        "健身房{}了{}小时，{}",
        "{}的综艺{}，笑得肚子疼",
        "打游戏{}，{}队友带飞",
    ],
}

# 每个场景的特色词汇（用于填槽）
SCENE_VOCAB = {
    0: {  # 购物消费
        "places": ["商场", "超市", "网店", "旗舰店", "直播间"],
        "items": ["衣服", "零食", "化妆品", "电器", "日用品", "鞋子"],
        "adj": ["划算", "实惠", "便宜", "满意", "不错", "一般"],
        "status": ["当天", "次日", "加急", "普通"],
        "platforms": ["淘宝", "京东", "拼多多", "抖音"],
    },
    1: {  # 学习教育
        "subjects": ["数学", "英语", "物理", "化学", "语文", "编程"],
        "difficulty": ["很难", "有点难", "不太难", "比较简单", "特别难"],
        "time": ["两个小时", "一下午", "一上午", "好几天"],
        "results": ["还不错", "很理想", "一般般", "超预期"],
        "feel": ["挺有意思", "有点枯燥", "很有用", "很有趣"],
    },
    2: {  # 旅行出行
        "places": ["云南", "北京", "上海", "三亚", "西藏", "日本", "泰国"],
        "scenes": ["风景", "美食", "民宿", "夜景", "海滩", "古镇"],
        "transport": ["高铁", "飞机", "自驾", "大巴"],
        "feel": ["很美", "震撼", "不错", "一般", "值得"],
        "duration": ["三天", "一周", "五天", "半个月"],
    },
    3: {  # 工作职场
        "projects": ["开发", "运营", "市场", "产品", "设计"],
        "status": ["完成了", "延期了", "启动了", "验收了"],
        "time": ["晚上十点", "凌晨", "八点", "周末"],
        "reaction": ["很满意", "不太满意", "表扬了", "批评了"],
        "years": ["三", "五", "两", "十"],
    },
    4: {  # 娱乐休闲
        "activities": ["宅家", "出去", "在线", "线下"],
        "games": ["王者", "原神", "吃鸡", "LOL"],
        "content": ["科幻", "喜剧", "悬疑", "爱情"],
        "time": ["凌晨两点", "半夜", "很晚"],
        "feel": ["很精彩", "很感人", "很搞笑", "很刺激"],
    },
}


def fill_template(template: str, vocab: dict) -> str:
    """用词汇填槽，让句子更自然"""
    result = template
    slots = {"places", "items", "adj", "status", "platforms",
             "subjects", "difficulty", "time", "results", "feel",
             "scenes", "transport", "duration", "projects", "reaction", "years",
             "activities", "games", "content"}

    for _ in range(template.count("{}")):
        # 随机选择一个可用的词填入
        available = [s for s in slots if s in vocab and vocab[s]]
        if available:
            slot = random.choice(available)
            word = random.choice(vocab[slot])
            result = result.replace("{}", word, 1)

    return result


def add_noise(sentence: str, noise_prob:0.2) -> str:
    """添加噪声：随机插入无关词，增强模型鲁棒性"""
    noise_words = ["真的", "感觉", "有点", "非常", "比较", "还算", "稍微", "挺"]
    if random.random() < noise_prob:
        pos = random.randint(0, len(sentence))
        word = random.choice(noise_words)
        sentence = sentence[:pos] + word + sentence[pos:]
    return sentence


def generate_sample(label: int) -> str:
    """生成指定类别的样本"""
    template = random.choice(SCENE_TEMPLATES[label])
    vocab = SCENE_VOCAB[label]
    sentence = fill_template(template, vocab)
    sentence = add_noise(sentence, noise_prob=0.3)
    return sentence


def build_dataset(n_per_class: int = 1000):
    """构建平衡数据集"""
    data = []
    for label in range(N_CLASSES):
        for _ in range(n_per_class):
            sentence = generate_sample(label)
            data.append((sentence, label))
    random.shuffle(data)
    return data


# ─── 2. 词表与编码 ──────────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset ─────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 多分类用long
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
models = {
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
}
class TextTorchModel(nn.Module):
    """基础 RNN 分类器"""
    def __init__(self, model_type, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = models[model_type](embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        e, _ = self.encoder(self.embedding(x))       # (B, L, H)
        pooled = e.max(dim=1)[0]                  # (B, H)
        out = self.fc(self.dropout(self.bn(pooled)))
        return out  # CrossEntropyLoss 内置 Softmax，不需要额外激活


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train_model(model_name, train_loader, val_loader, vocab_size, device):
    """训练单个模型"""
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    print(f"{'='*50}")

    model = TextTorchModel(model_name, vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"最佳验证准确率: {best_acc:.4f}")
    return model, best_acc


def inference_demo(models, vocab, device):
    """推理演示"""
    test_sents = [
        "刚在直播间买了一堆零食，价格挺划算的",
        "数学考试太难了，复习了一整天",
        "去云南旅游了一周，风景真的很美",
        "项目终于验收通过了，领导很满意",
        "周末和朋友打游戏，玩得很开心",
        "这道物理题有点难，花了好久才做出来",
        "淘宝买的衣服质量不错，下次还来",
        "坐高铁去了趟北京，只用了四个小时",
    ]

    label_names = ["购物消费", "学习教育", "旅行出行", "工作职场", "娱乐休闲"]

    print("\n" + "="*70)
    print("推理测试对比")
    print("="*70)

    for sent in test_sents:
        ids = torch.tensor([encode(sent, vocab)], dtype=torch.long).to(device)
        print(f"\n输入: {sent}")

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                logits = model(ids)
                prob = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).item()
                confidence = prob[0, pred].item()
            print(f"  [{name}] {label_names[pred]} (置信度: {confidence:.2f})")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 生成数据
    print("\n生成数据集...")
    data = build_dataset(n_per_class=N_SAMPLES // N_CLASSES)
    vocab = build_vocab(data)
    print(f"样本数: {len(data)}, 词表大小: {len(vocab)}")

    # 划分数据集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 训练 RNN
    rnn_model, rnn_acc = train_model(
        "RNN", train_loader, val_loader, len(vocab), device
    )

    # 训练 LSTM
    lstm_model, lstm_acc = train_model(
        "LSTM", train_loader, val_loader, len(vocab), device
    )

    # 对比结果
    print("\n" + "="*50)
    print("实验结果对比")
    print("="*50)
    print(f"RNN  最佳准确率: {rnn_acc:.4f}")
    print(f"LSTM 最佳准确率: {lstm_acc:.4f}")
    print(f"LSTM 相比 RNN 提升: {(lstm_acc - rnn_acc)*100:.2f}%")

    # 推理演示
    models = {"RNN": rnn_model, "LSTM": lstm_model}
    inference_demo(models, vocab, device)


if __name__ == '__main__':
    main()
