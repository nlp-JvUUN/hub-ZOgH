"""
GRPO 强化学习后全量模型交互对话脚本
用法: python src/chat_with_model.py [--ckpt outputs/grpo_ckpt]
默认加载强化学习后的全量 checkpoint，进行多轮对话，Ctrl+C 退出。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import io

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 与训练时一致的系统提示（算术助手，输出 <answer> 标签）
SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="outputs/grpo_ckpt",
                    help="模型 checkpoint 目录，默认强化学习后的全量模型")
parser.add_argument("--max_new_tokens", type=int, default=64)
args = parser.parse_args()

print(f"加载模型：{args.ckpt} ...")
tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 全量 ckpt 直接加载（显式 bf16，避免 fp16 下溢训废）
model = AutoModelForCausalLM.from_pretrained(
    args.ckpt, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("模型加载完成，开始对话（输入 q 或 Ctrl+C 退出）：\n")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user = input("你> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见。")
        break
    if not user:
        continue
    if user.lower() in ("q", "quit", "exit"):
        print("再见。")
        break

    messages.append({"role": "user", "content": user})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    resp = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print("AI>", resp, "\n")
    messages.append({"role": "assistant", "content": resp})