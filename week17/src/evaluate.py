"""在固定随机测试集上评估模型的数学正确率、格式率和 pass@k。"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_utils import LEVELS, make_problem, messages_for, parse_answer

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="评估算术模型")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "evaluation.json"))
    parser.add_argument("--n-per-level", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="每个难度只评估 5 道题")
    return parser.parse_args()


def format_prompts(tokenizer, problems):
    return [
        tokenizer.apply_chat_template(messages_for(expr), tokenize=False, add_generation_prompt=True)
        for expr, _ in problems
    ]


@torch.inference_mode()
def generate(model, tokenizer, prompts, sample: bool, k: int, batch_size: int):
    grouped = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **encoded,
            max_new_tokens=64,
            do_sample=sample,
            temperature=1.0 if sample else None,
            top_p=1.0 if sample else None,
            num_return_sequences=k if sample else 1,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = outputs[:, encoded["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if sample:
            grouped.extend(decoded[i * k:(i + 1) * k] for i in range(len(batch)))
        else:
            grouped.extend([[text] for text in decoded])
    return grouped


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("模型评估默认需要 CUDA GPU。")
    n = 5 if args.quick else args.n_per_level
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    rng, report = random.Random(args.seed), {}
    for level in LEVELS:
        problems = [make_problem(level, rng) for _ in range(n)]
        prompts = format_prompts(tokenizer, problems)
        greedy = generate(model, tokenizer, prompts, False, 1, args.batch_size)
        samples = generate(model, tokenizer, prompts, True, args.k, args.batch_size)
        greedy_fmt = greedy_correct = pass_k = informative = 0
        examples = []
        for (expr, answer), greedy_group, sample_group in zip(problems, greedy, samples):
            fmt, _, correct = parse_answer(greedy_group[0], answer)
            outcomes = [parse_answer(text, answer)[2] for text in sample_group]
            greedy_fmt += fmt
            greedy_correct += correct
            pass_k += any(outcomes)
            informative += 0 < sum(outcomes) < args.k
            if len(examples) < 3:
                examples.append({"expression": expr, "answer": answer, "output": greedy_group[0]})
        report[level] = {
            "n": n,
            "greedy_format_rate": round(greedy_fmt / n, 4),
            "greedy_accuracy": round(greedy_correct / n, 4),
            f"pass@{args.k}": round(pass_k / n, 4),
            "informative_group_rate": round(informative / n, 4),
            "examples": examples,
        }
        print(level, report[level])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评估结果已保存：{output}")


if __name__ == "__main__":
    main()
