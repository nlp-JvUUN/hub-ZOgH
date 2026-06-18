import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class SequenceLabelingDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        with open(file_path, 'r', encoding='utf - 8') as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example['text']
        label = example['label'].split()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label_ids = self.convert_label_to_ids(label, input_ids)
        return input_ids, attention_mask, label_ids

    def convert_label_to_ids(self, label, input_ids):
        label_ids = []
        subwords_count = 0
        for i, token_id in enumerate(input_ids):
            if token_id in self.tokenizer.all_special_ids:
                label_ids.append(-100)
            else:
                label_ids.append(self.label_to_id[label[subwords_count]])
                subwords_count += 1
        return torch.tensor(label_ids)

    label_to_id = {'O': 0, 'B - ENTITY': 1, 'I - ENTITY': 2}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = SequenceLabelingDataset('train.json', tokenizer)
train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first = True)

    def forward(self, input_ids, attention_mask, labels = None):
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        emissions = self.fc(lstm_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask = attention_mask.byte())
            return loss
        else:
            paths = self.crf.decode(emissions, mask = attention_mask.byte())
            return paths


model = BiLSTMCRF(
    vocab_size = len(tokenizer.vocab),
    embedding_dim = 128,
    hidden_dim = 256,
    num_tags = len(train_dataset.label_to_id)
)



import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, label_ids in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, label_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
