训练过程 (60 步):
  - reward 起始 ~0,中段峰值 0.665,末段 ~0.465
  - train_loss: 0.028
  - entropy: 0.69 → 0.36 (变自信)
  - 峰值显存: ~2.6 GB / 8 GB
  - 总耗时: 14.5 min

最终测试集 (130 条):
  - base    : 45/130 = 34.62%
  - trained : 43/130 = 33.08%
  - delta   : -1.54% (≈ 无变化)

"""在 GSM8K 测试子集上跑 base 或 trained(adapter)模型,报告 accuracy。

用法:
  python eval/eval_gsm8k.py --base                       # 评估 Qwen2.5-0.5B-Instruct 原模型
  python eval/eval_gsm8k.py --adapter outputs/grpo/final # 评估 LoRA 适配后模型
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import sys
from pathlib import Path

# 让脚本能找到 ../rewards
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.math_reward import extract_answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", action="store_true", help="评估原模型")
    p.add_argument("--adapter", default=None, help="LoRA adapter 路径")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--test_file", default="data/gsm8k_test.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0 表示用全部,否则取前 N 条")
    p.add_argument("--out", default=None, help="可选:把每条预测写到这里")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.base and not args.adapter:
        raise SystemExit("必须传 --base 或 --adapter")

    print(f"[load] base={args.base_model}, adapter={args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} test samples")

    correct = 0
    predictions = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i: i + args.batch_size]
        prompts = [r["prompt"] for r in batch]
        golds = [r["answer"] for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for t, gold in zip(texts, golds):
            pred = extract_answer(t)
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-3
            correct += int(ok)
            predictions.append({"gold": gold, "pred_text": t, "pred": pred, "ok": ok})
        print(f"  [{i+len(batch)}/{len(rows)}] acc so far = {correct/(i+len(batch)):.3f}")

    acc = correct / len(rows)
    print(f"\n[result] accuracy = {acc:.4f}  ({correct}/{len(rows)})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for p_ in predictions:
                f.write(json.dumps(p_, ensure_ascii=False) + "\n")
        print(f"[save] -> {args.out}")


if __name__ == "__main__":
    main()



  
