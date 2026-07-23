"""
BERT 文本分类模型 —— 支持三种微调策略

  1. full   : 全量微调（BERT + 分类头，全部参数参与训练）
  2. freeze : 冻结 BERT，仅训练分类头（feature-based）
  3. lora   : LoRA 高效微调（训练低秩适配器，参数量仅 ~0.3%）

实现要点：
  - 取 [CLS] token 表示整句语义，经分类头映射到类别
  - 一个模型类 BertCls 统一支持 3 种训练策略（full / freeze / lora）
  - 使用 HuggingFace AutoModel 加载预训练权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BertCls(nn.Module):
    """BERT 文本分类器。

    前向过程：
      input_ids → BertModel → pooler_output / [CLS] (B, 768)
               → Dropout → Linear(768, num_classes) → logits
    """

    def __init__(
            self,
            model_name: str = "bert-base-chinese",
            num_classes: int = 2,
            dropout: float = 0.1,
            freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self._freeze_bert = freeze_bert

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # 优先用 pooler_output，否则取 [CLS] hidden state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]  # [CLS] token

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    @property
    def trainable_params_ratio(self) -> float:
        """返回当前可训练参数占比（用于日志）。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable / total


def _inject_lora(model: BertCls, lora_r: int = 8, lora_alpha: float = 16.0) -> BertCls:
    """对内部 BERT 编码器的 query/value 投影矩阵注入 LoRA 低秩适配器。

    注意：只对 model.bert（AutoModel）注入 LoRA，而非整个 BertCls 包装器。
    这样 BertCls.forward() 签名保持不变，分类头参数也不被 LoRA 化。
    """
    from peft import get_peft_model, LoraConfig, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
    )
    # 只给内部 BERT 注入 LoRA，外层 BertCls 保持不变
    model.bert = get_peft_model(model.bert, lora_config)
    return model


def build_classifier(
        method: str = "full",
        model_name: str = "bert-base-chinese",
        num_classes: int = 2,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
) -> nn.Module:
    """模型工厂函数。

    method 取值：
      - "full"   : 全量微调，BERT + 分类头全部可训练
      - "freeze" : 冻结 BERT，仅训练分类头 + dropout
      - "lora"   : LoRA 高效微调，注入低秩适配器
    """
    freeze_bert = (method == "freeze")

    model = BertCls(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_bert=freeze_bert,
    )

    if method == "lora":
        try:
            model = _inject_lora(model, lora_r=lora_r, lora_alpha=lora_alpha)
        except ImportError:
            print("  ⚠ peft 未安装，回退到 full 模式（pip install peft）")
            return build_classifier("full", model_name, num_classes, dropout)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tag = {"full": "全量微调", "freeze": "冻结 BERT", "lora": "LoRA 微调"}[method]
    print(f"  模型：BERT + 分类头（{tag}）")
    print(f"  类别数：{num_classes}")
    print(f"  参数总量：{total / 1e6:.1f}M")
    print(f"  可训练参数：{trainable / 1e6:.1f}M（{trainable / max(total, 1) * 100:.2f}%）")

    return model
