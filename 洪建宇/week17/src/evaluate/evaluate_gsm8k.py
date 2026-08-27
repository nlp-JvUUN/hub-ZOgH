"""评测模型在 GSM8K 上的准确率，用于对比 GRPO 训练前后能力。

示例（对比前后）：
  python evaluate_gsm8k.py --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct --num_eval 200
  python evaluate_gsm8k.py --model_name_or_path outputs/step_50 --num_eval 200
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_math.data import PROMPT_TEMPLATE, extract_gsm8k_answer
from grpo_math.rewards import extract_model_answer, answer_equal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype
    ).to(dev).eval()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n = min(args.num_eval, len(ds))
    correct = 0
    for i in range(n):
        ex = ds[i]
        prompt = tok.apply_chat_template(
            [{"role": "user",
              "content": PROMPT_TEMPLATE.format(problem=ex["question"])}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        gt = extract_gsm8k_answer(ex["answer"])
        pred = extract_model_answer(text)
        if pred is not None and answer_equal(pred, gt):
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}  acc={correct/(i+1):.3f}")

    print(f"\nAccuracy on GSM8K ({args.split}): {correct}/{n} = {correct/n:.3f}")


if __name__ == "__main__":
    main()
