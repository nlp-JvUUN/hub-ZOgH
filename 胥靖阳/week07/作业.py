# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/6/4 19:27  
@脚本说明 : 

"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# ------------------------------
# 1. 数据加载与标签编码
# ------------------------------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_labels(data):
    label_set = set()
    for sample in data:
        for tag in sample['ner_tags']:
            if tag != 'O':
                label_set.add(tag)
    # 添加 'O' 标签
    label_set.add('O')
    label_list = sorted(list(label_set))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

# ------------------------------
# 2. 对齐子词标签（处理 BERT 分词导致的长度变化）
# ------------------------------
def align_labels_with_tokens(labels, tokenized_input, tokenizer):
    """
    将原始 token 级别的标签映射到 subword 级别。
    labels: 原始 token 对应的标签id列表，长度等于原 tokens 数量
    tokenized_input: tokenizer 输出，包含 'input_ids', 'offset_mapping' 等
    tokenizer: 用于获取词汇表等（这里不需要，仅示意）
    """
    aligned_labels = []
    offset_mapping = tokenized_input['offset_mapping']
    # 忽略特殊 token [CLS] 和 [SEP]
    # 遍历 offset_mapping，第一个和最后一个一般是特殊 token，忽略
    prev_token_idx = 0
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            # 特殊 token，标签设为 -100 忽略损失
            aligned_labels.append(-100)
            continue
        # 找到当前 subword 对应的原始 token 索引
        # 根据起始字符位置，找到对应的原始 token（假设每个原始 token 占据一个字符）
        # 由于原 tokens 是字符列表，而 offset 是字符索引，直接映射
        # 注意：可能有一个原始 token 被拆分成多个 subword，此时需要将标签赋予第一个 subword，后续 subword 用 -100
        # 简单起见，将第一个 subword 赋予原标签，后续忽略
        # 实现方法：检查当前 subword 是否属于新的原始 token
        while prev_token_idx < len(labels) and (start >= end_of_prev_token(offset_mapping, prev_token_idx)):
            prev_token_idx += 1
        if prev_token_idx < len(labels):
            aligned_labels.append(labels[prev_token_idx])
        else:
            aligned_labels.append(-100)
    return aligned_labels

def end_of_prev_token(offset_mapping, token_idx):
    """辅助函数，获取第 token_idx 个 token 的结束位置"""
    if token_idx == 0:
        return 0
    # 这里简化：假设原始 tokens 是单字符，offset 长度可能大于原始字符数，但通常每个原始字符对应一个 subword
    # 更精确的方法是通过累积每个原始 token 的字符长度
    # 但为了代码简洁，我们直接在外部处理
    pass

# 更稳健的映射方法：利用原始 tokens 的字符位置
def align_labels(tokens, labels, tokenizer_output):
    """
    tokens: 原始 token 列表（字符列表）
    labels: 原始标签列表（整数）
    tokenizer_output: tokenizer 编码后的结果，包含 input_ids, offset_mapping
    """
    # 构建字符到原始 token 索引的映射（每个字符属于哪个 token）
    char_to_token_idx = []
    for i, tok in enumerate(tokens):
        char_to_token_idx.extend([i] * len(tok))  # 每个 token 可能有多个字符？中文 tokens 是一个字符，所以 len=1
    # 但有的 token 可能是标点或数字，仍然是一个字符，所以没问题
    aligned_labels = []
    for idx, (start, end) in enumerate(tokenizer_output['offset_mapping'][1:-1]):  # 跳过 [CLS] 和 [SEP]
        if start == 0 and end == 0:
            aligned_labels.append(-100)
            continue
        # 获取该 subword 覆盖的字符区间内的字符，取第一个字符所属的原始 token 索引
        # 由于 subword 可能覆盖多个字符（罕见），简单取起始位置
        char_pos = start
        if char_pos < len(char_to_token_idx):
            token_idx = char_to_token_idx[char_pos]
            aligned_labels.append(labels[token_idx])
        else:
            aligned_labels.append(-100)
    # 添加 [CLS] 和 [SEP] 的标签
    aligned_labels = [-100] + aligned_labels + [-100]  # [CLS] 和 [SEP] 忽略
    return aligned_labels

# ------------------------------
# 3. 数据集类
# ------------------------------
class NERDataset(Dataset):
    def __init__(self, data, label2id, tokenizer, max_length=128):
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        tags = [self.label2id[tag] for tag in sample['ner_tags']]  # 原始标签 id

        # 将 tokens 拼接成完整句子（原始 token 是字符，直接拼接）
        text = ''.join(tokens)
        # 编码并获取 offset mapping
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        # 删除 batch 维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        # 对齐标签
        labels_aligned = align_labels(tokens, tags, encoding)
        # 如果对齐后的标签长度超过 max_length，则截断
        if len(labels_aligned) > self.max_length:
            labels_aligned = labels_aligned[:self.max_length]
        # 创建标签张量，用 -100 填充不足部分
        labels_tensor = torch.full((self.max_length,), -100, dtype=torch.long)
        labels_tensor[:len(labels_aligned)] = torch.tensor(labels_aligned, dtype=torch.long)

        # 移除 offset_mapping 字段（不传给模型）
        encoding.pop('offset_mapping')
        encoding['labels'] = labels_tensor
        return encoding

# ------------------------------
# 4. 评估函数（使用 seqeval）
# ------------------------------
def evaluate(model, dataloader, id2label, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # 过滤掉 -100 的标签
        for pred, label in zip(predictions.cpu().numpy(), batch['labels'].cpu().numpy()):
            valid_mask = label != -100
            pred_valid = pred[valid_mask]
            label_valid = label[valid_mask]
            all_preds.extend(pred_valid)
            all_labels.extend(label_valid)
    # 转换为标签字符串
    pred_str = [id2label[p] for p in all_preds]
    label_str = [id2label[l] for l in all_labels]
    report = classification_report(label_str, pred_str, digits=4)
    acc = accuracy_score(label_str, pred_str)
    return acc, report

# ------------------------------
# 5. 主训练流程
# ------------------------------
def train():
    # 配置
    data_path = 'validation.json'  # 请根据实际路径修改
    model_name = 'bert-base-chinese'
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    max_length = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './ner_model'

    # 加载数据
    print("Loading data...")
    data = load_data(data_path)
    label2id, id2label = extract_labels(data)
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {label2id}")

    # 分割训练/验证集
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    # 初始化 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # 创建数据集和 DataLoader
    train_dataset = NERDataset(train_data, label2id, tokenizer, max_length)
    val_dataset = NERDataset(val_data, label2id, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

        # 验证
        acc, report = evaluate(model, val_loader, id2label, device)
        print(f"Validation accuracy: {acc:.4f}")
        print(report)

        # 保存模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    print("Training finished.")

if __name__ == '__main__':
    train()