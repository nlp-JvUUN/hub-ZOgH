import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen2‑1.5B‑Instruct"
LORA_PATH = "./grpo‑math‑lora‑final"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)

prompt_text = """请一步步思考，解下面这道数学题：
问题：A买5个苹果花20元，买8个需要多少钱？
解题过程："""

inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))