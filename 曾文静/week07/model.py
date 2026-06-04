# -*- coding: utf-8 -*-
"""
week07/model.py
===============
BERT 序列标注模型, 两种解码头:
  - BERT + Linear: 逐 token 独立预测 (基线)
  - BERT + CRF:    条件随机场全局解码, 保证 BIO 序列合法
"""
import torch
import torch.nn as nn
from transformers import BertModel


class NERModel(nn.Module):
    def __init__(self, bert_name="bert-base-chinese", num_labels=21,
                 use_crf=False, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        labels: Linear 模式传 -100 掩码标签; CRF 模式传完整标签(填充位填 0)
        返回 (loss, decoded)
          - Linear: decoded 是 (B, L) 的 argmax 索引
          - CRF:    decoded 是 list[list[int]] 的 Viterbi 解码结果
        """
        hidden = self.bert(input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))   # (B, L, K)

        if self.use_crf:
            mask = attention_mask.bool()
            if labels is not None:
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            else:
                loss = None
            decoded = self.crf.decode(emissions, mask=mask)
            return loss, decoded

        if labels is not None:
            loss = nn.functional.cross_entropy(
                emissions.view(-1, emissions.size(-1)),
                labels.view(-1), ignore_index=-100)
        else:
            loss = None
        decoded = emissions.argmax(dim=-1)
        return loss, decoded
