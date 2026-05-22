import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from transformer_interview import TransformerEncoder

# ─────────────────────── 模型定义 ───────────────────────
# 与 language_model.py 中的 LM 保持完全一致，以便正确加载 checkpoint

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, n_head=8, ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.model_type = model_type
        # 兼容一个transformer的模型并且使用mask 下三角矩阵
        if model_type == "transformer":
            self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim))  # 位置编码
            self.transformer = TransformerEncoder(hidden=embed_dim, n_layer=num_layers, n_head=n_head, ff=ff)
            self.fc = nn.Linear(embed_dim, vocab_size)
        else:
            rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.rnn = rnn_cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(embed_dim, vocab_size)
    def generate_causal_mask(self, seq_len, device):
        """生成下三角矩阵 mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask  # True 表示需要被 mask 掉的位置
    
    def forward(self, x):
        _, T = x.shape
        e = self.embed(x)

        # -------------------- Transformer 前向 --------------------
        if self.model_type == "transformer":
            e = e + self.pos_embed[:, :T, :]
            e = self.drop(e)
            mask = self.generate_causal_mask(T, x.device)
            out = self.transformer(e, mask)

        # -------------------- LSTM / RNN 前向 --------------------
        else:
            e = self.drop(e)
            out, _ = self.rnn(e)
        # 输出层
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits

# ==================== 采样策略 ====================
def top_p_sampling(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    indices_to_remove = cumulative_probs > top_p
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = 0
    sorted_logits[indices_to_remove] = -float('inf')
    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, next_token).squeeze(-1)

def beam_search_generate(model, start_ids, vocab_size, device, gen_len=60, beam_size=3):
    model.eval()
    beams = [(start_ids[0].tolist(), 0.0)]
    with torch.no_grad():
        for _ in range(gen_len):
            new_beams = []
            for seq, score in beams:
                input_ids = torch.tensor([seq], device=device)
                logits = model(input_ids)[0, -1]
                log_probs = F.log_softmax(logits, -1)
                top_v, top_idx = log_probs.topk(beam_size)
                for v, idx in zip(top_v, top_idx):
                    new_beams.append((seq + [idx.item()], score + v.item()))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
    return torch.tensor([beams[0][0]])

def top_p_generate(model, start_ids, device, gen_len=60, top_p=0.9):
    model.eval()
    gen = start_ids.to(device)
    with torch.no_grad():
        for _ in range(gen_len):
            logits = model(gen)[:, -1]
            next_id = top_p_sampling(logits, top_p)
            gen = torch.cat([gen, next_id.unsqueeze(0)], dim=1)
    return gen

# ==================== 生成句子 ====================
def main():
    # 获取当前路径赋值给pathDir
    pathDir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=f"{pathDir}/best_model.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从 checkpoint 中恢复词表和模型超参
    ckpt     = torch.load(args.model_path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]
    model = LM(
        vocab_size=len(char2idx),
        embed_dim  = cfg["embed_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        model_type = cfg["model"],
        dropout = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("=" * 52)
    print("  请输入开始文字：")
    print("  exit = 退出")
    print("=" * 52)
    while True:
        print()
        start_str = input("  开始文字：")
        if not start_str:
            continue
        if start_str == "exit":
            print("退出。")
            break

        start_ids = torch.tensor([[char2idx[c] for c in start_str if c in char2idx]])

        print("\n===== Beam Search 束搜索生成 ======")
        beam_out = beam_search_generate(model, start_ids, len(char2idx), device, gen_len=60)
        print("".join([idx2char[i.item()] for i in beam_out[0]]))

        print("\n===== Top-P 采样生成 ======")
        topp_out = top_p_generate(model, start_ids, device, gen_len=60, top_p=0.9)
        print("".join([idx2char[i.item()] for i in topp_out[0]]))

if __name__ == "__main__":
    main()