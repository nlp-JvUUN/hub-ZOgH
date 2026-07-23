"""
生成脚本说明：
- 从训练好的 Transformer 单向语言模型检查点加载模型。
- 使用 beam search 生成文本，可指定 prompt、生成长度、温度和 beam width。

参数选择建议：
- temperature: 1.0 为默认平衡值，<1.0 更保守，>1.0 更随机。
- beam_width: 3-5 适合质量与速度折中，较大值会提高生成质量但更慢。
- length: 根据需求设置生成字符数，避免过长导致重复。

使用示例：
python generate.py --checkpoint test_model.pt --prompt "小时候" --length 200 --temperature 1.0 --beam_width 5
"""

import argparse
import torch
import torch.nn.functional as F
from 第五周作业 import TransformerLM


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    args = checkpoint["args"]
    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=args["embed_dim"],
        nhead=args.get("num_heads", 4),
        dim_feedforward=args.get("hidden_dim", 512),
        num_layers=args["num_layers"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, char2idx, idx2char


def beam_search(model, prompt_ids, length, temperature, beam_width, device):
    beams = [(prompt_ids, 0.0)]

    for _ in range(length):
        next_beams = []
        for seq, score in beams:
            input_ids = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_ids)
            log_probs = F.log_softmax(logits[0, -1] / max(temperature, 1e-6), dim=-1)
            topk = torch.topk(log_probs, beam_width)
            for token_id, token_logp in zip(topk.indices.tolist(), topk.values.tolist()):
                next_beams.append((seq + [token_id], score + token_logp))

        beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:beam_width]

    return beams


def decode(ids, idx2char):
    return "".join(idx2char[i] for i in ids if i in idx2char)


def main():
    parser = argparse.ArgumentParser(description="从 Transformer 语言模型检查点生成文本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_beams", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)

    prompt_ids = [char2idx[c] for c in args.prompt if c in char2idx]
    if not prompt_ids and args.prompt:
        raise ValueError("prompt 中包含未登录词，请使用语料中出现的字符。")

    beams = beam_search(model, prompt_ids, args.length, args.temperature, args.beam_width, device)

    print("生成结果：")
    for rank, (seq, score) in enumerate(beams[: args.top_beams], start=1):
        text = decode(seq, idx2char)
        print(f"--- Beam {rank} (score={score:.2f}) ---")
        print(text)
        print()

    if len(beams) >= 2:
        print("提示：可通过增加 beam_width 或 temperature 查看更多多样化生成结果。")


if __name__ == "__main__":
    main()
