import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 超参数
SEED        = 42
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 32
EPOCHS      = 50
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)


# 1、数据集 - 扩充后的新闻数据
label_names = ["金融", "娱乐", "体育", "科技", "健康"]

data = [
    # 金融 0
    ("股市今天上涨了", 0),
    ("银行利率下降", 0),
    ("人民币汇率创新高", 0),
    ("上市公司财报发布", 0),
    ("基金净值增长", 0),
    ("房地产市场回暖", 0),
    ("保险业务增长迅速", 0),
    ("投资理财产品推荐", 0),
    ("央行降准释放流动性", 0),
    ("股票指数突破新高", 0),
    ("企业融资成本降低", 0),
    ("外汇储备增加", 0),
    ("债券收益率上升", 0),
    ("消费信贷规模扩大", 0),
    ("财富管理市场发展", 0),
    
    # 娱乐 1
    ("这部电影很好看", 1),
    ("演员表现非常出色", 1),
    ("歌手发布新专辑", 1),
    ("综艺节目收视率高", 1),
    ("电视剧热播中", 1),
    ("明星演唱会门票售罄", 1),
    ("颁奖典礼盛况空前", 1),
    ("导演新作即将上映", 1),
    ("网络剧播放量破亿", 1),
    ("偶像团体出道", 1),
    ("电影票房突破纪录", 1),
    ("音乐排行榜更新", 1),
    ("真人秀节目热播", 1),
    ("艺人签约新公司", 1),
    ("影视作品获奖", 1),
    
    # 体育 2
    ("足球比赛今晚开始", 2),
    ("球队赢得冠军", 2),
    ("篮球联赛激烈进行", 2),
    ("运动员刷新世界纪录", 2),
    ("奥运会筹备进行中", 2),
    ("网球公开赛开赛", 2),
    ("游泳比赛精彩纷呈", 2),
    ("乒乓球选手夺冠", 2),
    ("马拉松赛事圆满结束", 2),
    ("健身教练分享经验", 2),
    ("排球联赛战况激烈", 2),
    ("冰雪运动发展迅速", 2),
    ("电子竞技比赛开幕", 2),
    ("运动员康复训练", 2),
    ("体育产业发展报告", 2),
    
    # 科技 3
    ("新手机发布了", 3),
    ("人工智能发展迅速", 3),
    ("5G网络建设加速", 3),
    ("芯片技术取得突破", 3),
    ("云计算服务升级", 3),
    ("大数据应用普及", 3),
    ("区块链技术应用", 3),
    ("新能源汽车上市", 3),
    ("虚拟现实技术成熟", 3),
    ("量子计算机研发", 3),
    ("智能家居产品热销", 3),
    ("自动驾驶测试进行", 3),
    ("软件版本更新发布", 3),
    ("网络安全防护加强", 3),
    ("半导体产业发展", 3),
    
    # 健康 4
    ("多运动有益健康", 4),
    ("饮食要均衡", 4),
    ("疫苗接种全面展开", 4),
    ("中医养生知识普及", 4),
    ("心理健康重要性", 4),
    ("营养膳食指南发布", 4),
    ("睡眠质量改善方法", 4),
    ("疾病预防措施", 4),
    ("医疗技术进步", 4),
    ("保健品市场规范", 4),
    ("公共卫生防护", 4),
    ("慢性病管理指南", 4),
    ("健康生活方式倡导", 4),
    ("体检项目推荐", 4),
    ("医药研发进展", 4),
]

# 2、构件词表与编码
def build_vocab(data):
    vocab = {'<pad>':0, '<unk>':1}
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

# 3、dataset/dataloader
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)
        )

# 4、模型 - 基础三层结构: embedding -> RNN -> 全连接层
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# 5、训练与评估
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    accuracy = correct / total

    
    return accuracy


def predict_text(text, model, vocab):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([encode(text, vocab)], dtype=torch.long)
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0].numpy()
    
    print(f"\n预测文本: {text}")
    print(f"预测类别: {label_names[pred]}")
    print("各类别概率:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {probs[i]:.4f}")
    return pred, probs


def train():
    print("="*60)
    print(f"新闻文本分类训练开始")
    print(f"数据集大小: {len(data)}")
    print(f"类别数: {len(label_names)} {label_names}")
    print("="*60)

    vocab = build_vocab(data)
    print(f"词表大小: {len(vocab)}")
    
    # 数据集划分
    random.shuffle(data)
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
    
    # 数据加载器
    train_dataset = TextDataset(train_data, vocab)
    test_dataset = TextDataset(test_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = RNNClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数: {total_params:,}")
    print("-" * 60)

    # 训练循环
    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        train_correct = train_total = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        avg_loss = total_loss / len(train_dataset)
        train_acc = train_correct / train_total
        test_acc = evaluate(model, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Best: {best_acc:.4f}")
    
    print("-" * 60)
    print(f"\n最佳测试准确率: {best_acc:.4f}")
    
    evaluate(model, test_loader)
    
    return model, vocab


if __name__ == "__main__":
    model, vocab = train()
    
    print("\n" + "="*60)
    print("模型预测测试")
    print("="*60)
    test_texts = [
        "阿里巴巴股票大涨",
        "周杰伦新专辑发布",
        "中国女排获得冠军",
        "苹果发布新款iPhone",
        "新冠疫苗研发成功"
    ]
    for text in test_texts:
        predict_text(text, model, vocab)
