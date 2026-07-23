"""
BertNER（线性头）和 BertCRFNER（CRF头）两个模型

适配 dataset.py 的数据格式：
  - NerDataset 返回字典：{input_ids, attention_mask, token_type_ids, labels}
  - labels 中 -100 表示不参与 loss 的位置（特殊 token、非首子词、PAD）
  - 支持前向传播时直接传入 labels 字段（model(**batch) 解包调用）

教学重点：
  1. 线性头（BertNER）：每个 token 独立预测标签
     - 问题：softmax 的独立预测忽略标签间的依赖关系
     - 可能产生非法序列：B-PER 后接 I-ORG，I-PER 开头等

  2. CRF 层（BertCRFNER）：加入转移矩阵，全局最优解码
     - 转移矩阵学习"什么标签之后可以接什么标签"
     - Viterbi 算法保证输出合法序列，永远不会 B-PER 后接 I-ORG
     - 代价：训练时需要前向-后向算法，比线性头慢约 20~30%

  3. 两者区别的量化：evaluate.py 会统计非法序列数

依赖：
  pip install pytorch-crf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel
from pathlib import Path
from typing import Optional


def _load_bert(bert_path: str) -> BertModel:
    """加载 BERT 模型，临时抑制 transformers 的警告日志。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。

    适配 dataset.py：
      - forward 接受 labels=None，有 labels 时自动计算 cross_entropy loss
      - ignore_index=-100 与 dataset.py 的子词对齐策略一致：
        [CLS]/[SEP]/PAD 及非首子词位置标记为 -100，不参与 loss
      - 支持 model(**batch) 解包调用（dataset 返回的字典直接传入）

    前向过程：
      input_ids → BertModel → last_hidden_state (B, L, 768)
               → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)

    损失：CrossEntropy，ignore_index=-100 跳过特殊token和非首子词
    预测：argmax(logits, dim=-1)
    """

    def __init__(
        self,
        bert_path: str,
        num_labels: int,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        # 保存标签映射，便于推理时将 id 转回标签名
        self.label2id = label2id or {}
        self.id2label = id2label or {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，兼容 dataset.py 返回的字典格式。

        Parameters
        ----------
        input_ids : (B, L) token id 序列
        attention_mask : (B, L) 注意力掩码，1=有效，0=PAD
        token_type_ids : (B, L) 句子编号
        labels : (B, L) BIO 标签 id，-100 表示不参与 loss

        Returns
        -------
        logits : (B, L, num_labels) 每个位置的标签分数
        loss : 标量或 None（labels 为 None 时不计算）
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """贪心解码：argmax(logits, dim=-1)，返回每条序列的预测标签 id 列表。

        返回的列表长度等于 attention_mask 中有效位置的数量（不含 PAD），
        但包含 [CLS]/[SEP] 位置的预测（这些位置在训练时标为 -100 不参与 loss，
        推理时仍会输出预测值，调用方可自行过滤）。

        Returns
        -------
        list[list[int]] : batch 中每条序列的预测标签 id 列表
        """
        logits, _ = self.forward(input_ids, attention_mask, token_type_ids)
        pred_ids = logits.argmax(dim=-1)  # (B, L)

        # 按实际长度截断，去掉 PAD 部分
        results = []
        for i in range(pred_ids.size(0)):
            length = attention_mask[i].sum().item()
            results.append(pred_ids[i, :length].tolist())
        return results


class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    适配 dataset.py 的关键设计：
      - dataset.py 对非首子词、特殊 token、PAD 位置标记 -100
      - CRF 不支持 ignore_index，需要手动构造 loss_mask：
        loss_mask = (labels != -100) & attention_mask.bool()
        只在 loss_mask=True 的位置计算 CRF 转移概率
      - 这样非首子词位置不会影响 CRF 的转移约束学习

    与 BertNER 的区别：
      - Linear 输出称为 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加转移矩阵，用 Viterbi 找全局最优序列
      - 损失：负对数似然（CRF 内部计算前向-后向），只对有效标签位置计算
      - 解码：self.crf.decode() 返回保证合法的标签序列

    CRF 的约束（自动学习）：
      - 初始只能以 O 或 B-X 开头
      - B-X 之后只能是 I-X 或 B-Y 或 O
      - I-X 之后只能是 I-X 或 B-Y 或 O
    """

    def __init__(
        self,
        bert_path: str,
        num_labels: int,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torchcrf import CRF

        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

        # 保存标签映射，便于推理时将 id 转回标签名
        self.label2id = label2id or {}
        self.id2label = id2label or {}

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """获取 CRF 的发射分数（BERT 输出经过线性映射）。"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，兼容 dataset.py 返回的字典格式。

        CRF 损失计算的适配逻辑：
          1. dataset.py 中 labels=-100 的位置包括：
             - [CLS], [SEP], PAD 等特殊 token
             - 非首子词（同一汉字被 tokenizer 拆分后的续接子词）
          2. CRF 不支持 ignore_index，所以需要：
             - 将 -100 替换为合法标签 id（如 0，即 O）
             - 构造 loss_mask = (labels != -100) & attention_mask.bool()
             - 用 loss_mask 替代原始 attention_mask 传入 CRF
          3. 这样 CRF 只在有真实标签的位置（首子词）计算转移概率，
             非首子词和特殊 token 完全不参与 loss

        Parameters
        ----------
        input_ids : (B, L) token id 序列
        attention_mask : (B, L) 注意力掩码
        token_type_ids : (B, L) 句子编号
        labels : (B, L) BIO 标签 id，-100 表示不参与 loss

        Returns
        -------
        emissions : (B, L, num_labels) CRF 发射分数
        loss : 标量或 None（labels 为 None 时不计算）
        """
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)

        loss = None
        if labels is not None:
            # ── 构造 CRF 专用的 mask 和 labels ──
            # loss_mask: 只在有真实标签的位置（labels != -100 且 attention_mask=1）计算 loss
            # 这与 dataset.py 的子词对齐策略一致：
            #   首子词 → 保留 BIO 标签（参与 loss）
            #   非首子词 / 特殊 token → 标记 -100（不参与 loss）
            loss_mask = (labels != -100) & attention_mask.bool()

            # 将 -100 替换为合法标签 id（0 = O）
            # 替换后的值不会影响 loss，因为 loss_mask 中对应位置为 False
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0

            # crf() 返回对数似然（正值），取负得到损失
            # 使用 loss_mask 替代 attention_mask，确保只在有效标签位置计算转移概率
            loss = -self.crf(emissions, labels_crf, mask=loss_mask, reduction="mean")

        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi 解码，返回 list[list[int]]，每条序列长度等于实际token数（不含PAD）。

        注意：解码时使用完整的 attention_mask（而非 loss_mask），
        因为推理时所有有效 token 位置都需要输出预测标签，
        包括 [CLS]/[SEP] 和非首子词位置。
        调用方可根据 word_ids() 过滤非首子词位置。
        """
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_model(
    use_crf: bool,
    bert_path: str,
    num_labels: int,
    label2id: Optional[dict[str, int]] = None,
    id2label: Optional[dict[int, str]] = None,
    dropout: float = 0.1,
) -> nn.Module:
    """模型工厂函数，与 dataset.py 的 build_label_schema 配合使用。

    使用方式
    --------
    from dataset import build_label_schema

    labels, label2id, id2label = build_label_schema()
    model = build_model(
        use_crf=True,
        bert_path="bert-base-chinese",
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    Parameters
    ----------
    use_crf : bool
        True 使用 BertCRFNER，False 使用 BertNER
    bert_path : str
        BERT 预训练模型路径或名称
    num_labels : int
        标签数量
    label2id : dict[str, int], optional
        标签名 → id 映射（来自 build_label_schema）
    id2label : dict[int, str], optional
        id → 标签名 映射（来自 build_label_schema）
    dropout : float
        Dropout 概率，默认 0.1

    Returns
    -------
    model : nn.Module
        构建好的模型实例
    """
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(
        bert_path=bert_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print("=" * 50)
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    print("=" * 50)
    return model
