"""快速测试 dataset.py 的核心功能"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, r"d:\homework\week7 序列标注问题\week07作业")

from src.dataset import build_label_schema, NerDataset
from transformers import BertTokenizer

# 1. 测试 build_label_schema
labels, l2i, i2l = build_label_schema()

# 2. 加载 tokenizer
tok = BertTokenizer.from_pretrained("bert-base-chinese")

# 3. 构造测试样本（含实体）
rec = [
    {
        "tokens": ["胡", "老", "说", "，", "当", "画", "画", "疲", "倦", "时"],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O"],
    }
]

# 4. 构建 Dataset 并取一个样本
ds = NerDataset(rec, tok, l2i, max_length=16)
sample = ds[0]

print("\ninput_ids shape:", sample["input_ids"].shape)
print("labels shape:", sample["labels"].shape)
print("attention_mask shape:", sample["attention_mask"].shape)
print("token_type_ids shape:", sample["token_type_ids"].shape)

print("\n解码子词对齐:")
for i, (iid, lbl) in enumerate(
    zip(sample["input_ids"].tolist(), sample["labels"].tolist())
):
    tok_str = tok.convert_ids_to_tokens(iid)
    lbl_str = i2l.get(lbl, str(lbl)) if lbl != -100 else "-100"
    print(f"  [{i:>2d}] {tok_str:<8s} → label={lbl_str}")

# 5. 验证关键对齐逻辑
# [CLS] → -100
# 胡 → B-PER (1)
# 老 → I-PER (2)
# 说 → O (0)
# [SEP] → -100
# [PAD]... → -100
assert sample["labels"][0].item() == -100, "[CLS] 应该标记为 -100"
assert sample["labels"][1].item() == 1, "'胡' 应该标记为 B-PER (1)"
assert sample["labels"][2].item() == 2, "'老' 应该标记为 I-PER (2)"
assert sample["labels"][3].item() == 0, "'说' 应该标记为 O (0)"

# 所有 PAD 位置都应该是 -100
pad_positions = (sample["attention_mask"] == 0).nonzero(as_tuple=True)[0]
for pos in pad_positions:
    assert sample["labels"][pos].item() == -100, f"PAD 位置 {pos.item()} 应该标记为 -100"

print("\n✅ 所有断言通过！子词对齐逻辑正确。")
