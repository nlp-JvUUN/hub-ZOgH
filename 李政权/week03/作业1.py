import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# ======================
# 超参数
# ======================
EMBED_DIM = 64
HIDDEN_DIM = 64
SEED_NUM = 42
N_SAMPLES = 4000
LR = 1e-3
BATCH_SIZE = 64
MAX_LEN = 32
EPOCHS = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5   # 5 类

# 设置随机种子
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)

# 模板（标签从 0 开始）
TEMPLATES_POS = [
    ('你要去哪里', 0),
    ('对你不客气', 1),
    ('这是你的吗', 2),
    ('真的是你吗', 3),
    ('把它还给你', 4),
]

TEMPLATES_OTHER_POS = [
    '他',
    '姐',
    '弟',
    '妹',
    '夫'
]


# ======================
# RNN 模型
# ======================
class MyRnnModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        # x: [B, L]
        e, _ = self.rnn(self.embedding(x))  # [B, L, H]
        pooled = e.max(dim=1)[0]            # [B, H]
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)            # [B, 5]
        return logits


# ======================
# 数据构建
# ======================
def build_dataset(size=N_SAMPLES):
    return [random.choice(TEMPLATES_POS) for _ in range(size)]


def build_dataset1(size=N_SAMPLES):
    this_data = []
    for _ in range(size):
        text, text_id = random.choice(TEMPLATES_POS)
        if text_id % 3 == 0:
            text = text.replace('你', random.choice(TEMPLATES_OTHER_POS))
        this_data.append((text, text_id))

    return this_data


def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            # if ch not in TEMPLATES_OTHER_POS:
            vocab.setdefault(ch, len(vocab))
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in text]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ======================
# 评估
# ======================
def evaluate(this_model, loader):
    this_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = this_model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ======================
# 评估
# ======================
def evaluate_test(this_model, loader):
    this_model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = this_model(x)
            print(f"logits=={logits}")
            pred = logits.argmax(dim=1)
            print(f"pred = {pred} | y = {y}")
            print(f"pred=={pred}")
            ss = torch.tensor(pred[0]).item()
            print(f"ss=={ss}")


def save_model(this_model, vocab_size, model_path="myRnnModel.bin"):
    """保存模型（包含元数据）"""
    checkpoint = {
        'model_state_dict': this_model.state_dict(),
        'vocab_size': vocab_size,  # 你的模型输入词汇表大小
        'model_type': 'TorchModel'
    }
    torch.save(checkpoint, model_path)
    print(f"模型已保存为 '{model_path}'")


def load_model(model_path="myRnnModel.bin"):
    """加载训练好的模型"""
    try:
        # 加载保存的模型参数
        checkpoint = torch.load(model_path)

        # 检查保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 新格式：包含模型状态和元数据
            vocab_size = checkpoint['vocab_size']
            model_state_dict = checkpoint['model_state_dict']
        else:
            # 旧格式：只包含模型状态
            # 需要从模型结构推断input_size
            # 这里假设输入大小为00因为你的模型是固定的
            vocab_size = 20
            model_state_dict = checkpoint

        # 创建模型实例
        model = MyRnnModel(vocab_size)

        # 加载模型参数
        model.load_state_dict(model_state_dict)
        model.eval()  # 设置为评估模式

        print(f"模型已从 '{model_path}' 加载")
        return model
    except:
        return None


def predict_batch_input(test_sample_num):
    """使用训练好的模型进行批量预测"""
    model = load_model("myRnnModel.bin")
    if model is None:
        print("无法加载模型，请先训练模型")
        return
    test_text = ["你要去哪里", "这是你的吗", "对你不客气"]
    for text in test_text:
        data = [(text, 0)]
        vocab = build_vocab(data)
        test_ds = TextDataset(data, vocab)
        test_loader = DataLoader(test_ds)
        print(f"测试数据为：{text}")
        evaluate_test(model, test_loader)


    # evaluate_test(model, test_loader)
    # print(f"输入样本{vocab}个，准确率为{acc * 100:.2f}%")


# ======================
# 训练
# ======================
def train():
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    split = int(len(data) * TRAIN_RATIO)
    train_ds = TextDataset(data[:split], vocab)
    val_ds = TextDataset(data[split:], vocab)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = MyRnnModel(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Acc: {acc:.4f}")

    save_model(model, len(vocab) )

    return model, vocab


def inference(model, vocab, sentences):
    model.eval()
    results = []

    with torch.no_grad():
        for sent in sentences:
            x = torch.tensor(
                [encode(sent, vocab)],
                dtype=torch.long
            )
            logits = model(x)
            print(f"logits=={logits}")
            pred_id = logits.argmax(dim=1).item()

            results.append({
                "text": sent,
                "pred_id": pred_id
            })

    return results


# ======================
# 主入口
# ======================
if __name__ == '__main__':
    model, vocab = train()
    # model = load_model()
    # data = build_dataset(1000)
    # data1 = build_dataset1(10)
    # print(f"data={data}")
    # print(f"data1={data1}")
    # vocab = build_vocab(data)
    #
    # test_sentences = [
    #     "你要去哪里",
    #     "真的是你吗",
    #     "你要去哪啊",
    #     "你不客气啊",
    #     "把你还给他",
    #     "这是你的吗"
    # ]
    #
    # print("\n===== 样本测试结果 =====")
    # results = inference(model, vocab, test_sentences)
    #
    # for r in results:
    #     print(f"输入: {r['text']}")
    #     print(f"预测位置: 第{r['pred_id'] + 1}位")
    #     print("-" * 40)
