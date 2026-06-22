import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
#  用于加载模型
# （必须与训练时结构一致）
# =========================================================

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        # Token Embedding
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # pos Embedding  nn.Parameter（可学习的位置编码）
        # torch.randn 生成一个标准正态分布（均值0，方差1）的随机tensor  参数1:batch维度 512:最大长度  embed_dim: 每个位置的维度  也就是为512个位置，每个位置准备一个embed_dim维的可学习向量
        # nn.Parameter(torch.randn(1, 512, embed_dim))这句代码也可以改为用Embedding方式:
        # self.pos_embed = nn.Embedding(512, embed_dim)
        # positions = torch.arange(seq_len).unsqueeze(0).to(x.device)
        # 最后 x = x + self.pos_embed(positions)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim))
        self.segment_embed = nn.Embedding(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # 模型维度（输入/输出特征维度）
            nhead=4,  # 注意力头数（将embed_dim分成4个头） bert里默认是12个头  该小一点
            dim_feedforward=128,  # FFN 神经元
            dropout=dropout,  # Dropout比率（防止过拟合）
            batch_first=True,  # 输入形状为 (batch, seq, dim)，而非 (seq, batch, dim
            activation="gelu"  # 激活函数（GELU比ReLU更平滑）
        )
        # 2层 Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    # 生成下三角 Mask（Attention Mask）
    def attention_mask(self, sz, device):
        mask = torch.triu(
            torch.ones(sz, sz, device=device),
            diagonal=1
        ).bool()

        return mask

    def forward(self, x):
        B, seq_len = x.size()
        segment_ids = torch.zeros(
            (B, seq_len),
            dtype=torch.long,
            device=x.device
        )
        segment_embedding = self.segment_embed(segment_ids)
        e = self.embed(x) + self.pos_embed[:, :seq_len, :]+segment_embedding
        e = self.drop(e)
        # 下三角 Attention Mask
        mask = self.attention_mask(seq_len, x.device)
        out = self.transformer(
            e,
            mask=mask
        )
        logits = self.fc(self.drop(out))  # (B, T, V)
        return logits

# =========================================================
# 加载模型
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 从 checkpoint 中恢复词表和模型超参
ckpt = torch.load(
    "best_model_trans.pt",
    map_location=device
)

char2idx = ckpt["char2idx"]
idx2char = ckpt["idx2char"]
cfg = ckpt["args"]

model = LM(
    vocab_size=len(char2idx),
    embed_dim=cfg["embed_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    model_type=cfg["model"],
    dropout=cfg["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("模型加载完成")

print("\n开始文本生成")
# =========================================================
# 文本生成（Greedy Search）
# =========================================================

def greedy_style(model,start_text,max_new_tokens=100):
    # 文本 -> token ids
    input_ids = []
    for c in start_text:
        if c in char2idx:
            input_ids.append(char2idx[c])

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            print(f"logits的形状{logits.shape}")
            # 只取最后一个位置
            last_logits = logits[:, -1]
            print(f"last_logits:{last_logits}")
            # 贪心策略（Greedy Search）直接取概率最大的 token
            next_token = torch.argmax(
                last_logits,
                dim=-1
            )
            for i, idx in enumerate(next_token):
                token = idx2char[idx.item()]
                # print(f"Batch {i} predicted: '{token}'")
            # 拼接到输入后面
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(1)],
                dim=1
            )

    # ids -> text
    output_ids = input_ids[0].tolist()
    output_text = "".join(
        idx2char[i]
        for i in output_ids
    )
    return output_text

# 调用
text = greedy_style(
    model,
    start_text="今天天气",
    max_new_tokens=20
)


print("\n贪心生成结果：\n")
print(text)


