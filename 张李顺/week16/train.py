import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).parent
MODEL_PATH = Path(
    r"C:\Users\17654\.cache\modelscope\hub\models\Qwen\Qwen2-0___5B-Instruct"
)
GROUP_SIZE = 8
MAX_NEW_TOKENS = 12
TRAIN_SECONDS = 240
FORMAT = re.compile(r"^<hahaha>-?\d+<gagaga>$")

random.seed(16)
torch.manual_seed(16)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_prompt(question):
    messages = [
        {"role": "user", "content": "计算：2 + 3"},
        {"role": "assistant", "content": "<hahaha>5<gagaga>"},
        {"role": "user", "content": f"计算：{question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def reward(response, answer):
    format_score = 0.5 if FORMAT.fullmatch(response) else 0.0
    numbers = re.findall(r"-?\d+", response)
    answer_score = 1.0 if numbers and numbers[-1] == answer else 0.0
    return format_score + answer_score


def answer_question(question):
    inputs = tokenizer(make_prompt(question), return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


def evaluate(data, output_path):
    model.eval()
    results = []
    for item in data:
        response = answer_question(item["question"])
        results.append(
            {
                "question": item["question"],
                "target": item["target"],
                "response": response,
                "correct": response == item["target"],
            }
        )

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return statistics(results)


def statistics(results):
    numeric = 0
    formatted = 0
    exact = 0

    for item in results:
        answer = re.fullmatch(
            r"<hahaha>(-?\d+)<gagaga>", item["target"]
        ).group(1)
        numbers = re.findall(r"-?\d+", item["response"])
        numeric += bool(numbers) and numbers[-1] == answer
        formatted += bool(FORMAT.fullmatch(item["response"]))
        exact += item["response"] == item["target"]

    return {"total": len(results), "numeric": numeric, "format": formatted, "exact": exact}


def print_comparison(before, after):
    total = before["total"]
    print("\n最终结果")
    for label, key in (("数值正确", "numeric"), ("格式正确", "format"), ("严格正确", "exact")):
        before_rate = before[key] / total
        after_rate = after[key] / total
        print(
            f"{label}：训练前 {before[key]}/{total} ({before_rate:.1%})"
            f" → 训练后 {after[key]}/{total} ({after_rate:.1%})"
        )


def token_log_probs(sequences, prompt_length):
    logits = model(
        input_ids=sequences,
        attention_mask=torch.ones_like(sequences),
        use_cache=False,
    ).logits[:, :-1]
    next_tokens = sequences[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen_log_probs = log_probs.gather(2, next_tokens.unsqueeze(2)).squeeze(2)
    return chosen_log_probs[:, prompt_length - 1 :]


def train_one_question(item):
    inputs = tokenizer(make_prompt(item["question"]), return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[1]

    model.eval()
    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_return_sequences=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion_ids = sequences[:, prompt_length:]
    completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)
    for row, tokens in enumerate(completion_ids):
        eos_positions = (tokens == tokenizer.eos_token_id).nonzero()
        if len(eos_positions):
            completion_mask[row, eos_positions[0].item() + 1 :] = 0

    responses = [
        tokenizer.decode(
            tokens[completion_mask[row].bool()], skip_special_tokens=True
        )
        for row, tokens in enumerate(completion_ids)
    ]
    rewards = torch.tensor(
        [reward(response, item["answer"]) for response in responses],
        device="cuda",
    )
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)

    with torch.no_grad():
        old_log_probs = token_log_probs(sequences, prompt_length)

    model.train()
    for _ in range(2):
        new_log_probs = token_log_probs(sequences, prompt_length)
        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages[:, None]
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[:, None]
        loss = -(
            torch.minimum(unclipped, clipped) * completion_mask
        ).sum() / completion_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return rewards.mean().item(), loss.item()


assert torch.cuda.is_available(), "当前环境不能使用CUDA"
assert MODEL_PATH.exists(), f"找不到本地模型：{MODEL_PATH}"

train_data = load_json(ROOT / "data" / "train.json")
eval_data = load_json(ROOT / "data" / "eval.json")
output_dir = ROOT / "output"
output_dir.mkdir(exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).to("cuda")
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad), lr=5e-5
)

print("评估训练前模型...")
before_statistics = evaluate(eval_data, output_dir / "before.json")
print(
    f"训练前严格正确率："
    f"{before_statistics['exact'] / before_statistics['total']:.2%}"
)

random.shuffle(train_data)
started = time.perf_counter()
step = 0
while time.perf_counter() - started < TRAIN_SECONDS:
    mean_reward, loss = train_one_question(train_data[step % len(train_data)])
    step += 1
    if step % 5 == 0:
        elapsed = time.perf_counter() - started
        print(
            f"步骤 {step} | 奖励 {mean_reward:.2f} | "
            f"损失 {loss:.4f} | {elapsed:.0f}秒"
        )

training_time = time.perf_counter() - started
model.save_pretrained(output_dir / "qwen_math_lora")
tokenizer.save_pretrained(output_dir / "qwen_math_lora")

print("评估训练后模型...")
after_statistics = evaluate(eval_data, output_dir / "after.json")
print(f"训练耗时：{training_time:.1f}秒")
print_comparison(before_statistics, after_statistics)
