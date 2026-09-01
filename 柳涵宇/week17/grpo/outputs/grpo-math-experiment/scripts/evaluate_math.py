from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grpo_math.rewards import answers_equal, extract_answer


SYSTEM_PROMPT = (
    "You are a careful math solver. Show the reasoning briefly, then put only "
    "the final answer in \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate math-answer accuracy.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def make_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokenizer, model = load_model(args)
    correct = 0
    examples = []

    for idx, row in enumerate(dataset):
        question = row[args.question_column]
        gold = extract_answer(str(row[args.answer_column]))
        prompt = make_prompt(tokenizer, question)
        completion = generate_answer(tokenizer, model, prompt, args)
        pred = extract_answer(completion)
        ok = answers_equal(pred, gold)
        correct += int(ok)
        if len(examples) < 5:
            examples.append((idx, ok, pred, gold, completion[:400].replace("\n", " ")))
        print(f"{idx + 1}/{len(dataset)} correct={correct} pred={pred!r} gold={gold!r}")

    accuracy = correct / max(len(dataset), 1)
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(dataset)})")
    print("\nSample generations:")
    for idx, ok, pred, gold, text in examples:
        print(f"- #{idx}: ok={ok} pred={pred!r} gold={gold!r} text={text}")


if __name__ == "__main__":
    main()
