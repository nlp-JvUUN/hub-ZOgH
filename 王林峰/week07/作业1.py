import json
import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

# ===================== 配置项 =====================
local_model_path = r"E:\pythoncommit\hub-ZOgH\王林峰\week07\bert-base-chinese"
max_len = 128
batch_size = 16
lr = 2e-5
epochs = 3

# ===================== 1. 加载标签 =====================
with open('label_names.json', 'r', encoding='utf-8') as f:
    label_list = json.load(f)

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# ===================== 2. 加载数据 =====================
with open('validation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 划分训练集/验证集
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# ===================== 3. 分词器 =====================
tokenizer = BertTokenizerFast.from_pretrained(local_model_path)

# ===================== 4. 数据集类 =====================
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        raw_labels = item['ner_tags']
        text = "".join(tokens)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_offsets_mapping=True
        )

        offset_mapping = encoding['offset_mapping']
        aligned_labels = []

        for token_start, token_end in offset_mapping:
            if token_start == 0 and token_end == 0:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label2id[raw_labels[token_start]])

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# ===================== 5. 模型 =====================
model = BertForTokenClassification.from_pretrained(
    local_model_path,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# ===================== 6. 设备 =====================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# ===================== 7. 数据加载 =====================
train_dataset = NERDataset(train_data, tokenizer, label2id, max_len)
val_dataset = NERDataset(val_data, tokenizer, label2id, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ===================== 8. 优化器 =====================
optimizer = optim.AdamW(model.parameters(), lr=lr)

# ===================== 9. 训练函数 =====================
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # 兼容所有版本
        if isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

# ===================== 10. 评估 =====================
def evaluate(model, data_loader, device, id2label):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 兼容所有版本
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            for i in range(len(predictions)):
                pred_seq = []
                true_seq = []
                for j in range(len(predictions[i])):
                    if attention_mask[i][j] == 1 and labels[i][j] != -100:
                        pred_seq.append(id2label[predictions[i][j].item()])
                        true_seq.append(id2label[labels[i][j].item()])
                pred_labels.append(pred_seq)
                true_labels.append(true_seq)

    return classification_report(true_labels, pred_labels)

# ===================== 11. 运行 =====================
if __name__ == '__main__':
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"训练损失: {train_loss:.4f}")

        report = evaluate(model, val_loader, device, id2label)
        print("\n评估结果:")
        print(report)

        model.save_pretrained(f'./ner_model_epoch_{epoch+1}')
        tokenizer.save_pretrained(f'./ner_model_epoch_{epoch+1}')