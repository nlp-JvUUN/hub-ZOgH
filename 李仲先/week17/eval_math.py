import argparse
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen2.5-0.5B-Instruct"

THINK_OPEN = "\x3cthink\x3e"
THINK_CLOSE = "\x3c/think\x3e"

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Solve the problem step by step.\n"
    f"Put your full reasoning inside {THINK_OPEN}...{THINK_CLOSE} tags.\n"
    f"After the {THINK_CLOSE} tag, put ONLY the final numeric answer inside \\boxed{{}} "
    "(e.g. \\boxed{72}). Do not include units inside \\boxed{}."
)

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_prediction(text: str):
    matches = BOXED_PATTERN.findall(text)
    candidate = matches[-1] if matches else None
    if candidate is None:
        numbers = NUMBER_PATTERN.findall(text)
        if not numbers:
            return None
        candidate = numbers[-1]
    candidate = candidate.replace(",", "").replace("$", "").replace("%", "").strip()
    m = NUMBER_PATTERN.search(candidate)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_gold(answer: str):
    value = answer.split("####")[-1].strip().replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate math ability on GSM8K test")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter dir produced by train_grpo.py")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    test = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_samples < len(test):
        test = test.select(range(args.num_samples))

    correct = 0
    total = 0
    results = []

    for example in tqdm(test, desc="evaluating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        gold = extract_gold(example["answer"])
        pred = extract_prediction(completion)
        is_correct = gold is not None and pred is not None and abs(pred - gold) < 1e-3
        correct += int(is_correct)
        total += 1
        results.append({
            "question": example["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "completion": completion,
        })

    accuracy = correct / total
    print(f"accuracy: {correct}/{total} = {accuracy:.4f}")

    if args.save_path:
        import json
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"detailed results saved to {args.save_path}")


if __name__ == "__main__":
    main()
