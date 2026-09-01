import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import GRPOTrainer
from peft import LoraConfig

# ===================== 配置参数 =====================
MODEL_NAME = "Qwen2‑1.5B‑Instruct"  # 可替换 Qwen2‑7B‑Instruct / Llama3‑8B‑Instruct
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 512
NUM_GENERATIONS = 4  # GRPO每组采样4条回答，计算组内相对奖励
BATCH_SIZE = 2
GRPO_BETA = 0.04  # KL散度约束系数
LEARNING_RATE = 5e‑5

# LoRA配置，只微调部分注意力层
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ===================== 加载数据集 GSM8K 数学题 =====================
def format_sample(sample):
    """构造prompt：数学问题输入，不包含答案"""
    prompt = f"""请一步步思考，解下面这道数学题：
问题：{sample['question']}
解题过程："""
    return {"prompt": prompt, "ground_truth_answer": sample["answer"]}


dataset = load_dataset("gsm8k", "main", split="train[:5000]")
dataset = dataset.map(format_sample)

# ===================== 奖励函数：数学题答案匹配奖励 =====================
def reward_fn(samples, prompts, completions, **kwargs):
    """
    GRPO奖励函数
    samples: 原始数据集样本
    prompts: 输入prompt列表
    completions: 模型生成输出的字符串列表
    return list[float]，每个样本对应奖励分数 [0~1]
    """
    rewards = []
    for sample, completion in zip(samples, completions):
        gt_ans = sample["ground_truth_answer"]
        # gsm8k标准答案格式：#### 数字，提取真实数字
        try:
            true_num = gt_ans.split("####")[-1].strip()
        except:
            true_num = ""

        # 从模型输出里提取最后出现数字作为模型答案
        import re
        nums = re.findall(r"‑?\d+\.?\d*", completion)
        pred_num = nums[-1] if len(nums) > 0 else ""

        # 答案匹配得1分，不匹配0分；简单奖励，真实业务可以做更复杂打分
        if pred_num == true_num:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ===================== 加载模型、Tokenizer =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# ===================== GRPO训练参数 =====================
training_args = TrainingArguments(
    output_dir="./grpo‑math‑output",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    peft_config=lora_config,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_generations=NUM_GENERATIONS,
    beta=GRPO_BETA,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./grpo‑math‑lora‑final")
    print("GRPO训练完成，LoRA权重保存完毕")