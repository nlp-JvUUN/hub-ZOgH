import time
from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_path_one = "/path/to/model"
sampling_params_one = SamplingParams(
    temperature_one=0.7,
    max_tokens_one=512,
    top_p_one=0.9
)

llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# 测试prompt列表
prompts = [
    "写一篇200字科技短文",
    "解释什么是分页注意力",
    "介绍大模型推理优化方案",
    "如何提升大模型并发速度"
]

# 测速one
start_one = time.time()
outputs = llm.generate(prompts, sampling_params_one)
end_one = time.time()

total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
total_time = end_one - start_one
tps_one = total_tokens / total_time

print(f"总耗时：{total_time:.2f}s")
print(f"总生成token：{total_tokens}")
print(f"平均吞吐：{tps_one:.2f} token/s")

model_path = "/path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
start_two = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7
    )
end_two = time.time()

gen_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
cost = end_two - start_two
tps_base = gen_tokens / cost

print(f"原生transformers 耗时：{cost:.2f}s")
print(f"吞吐：{tps_base:.2f} token/s")
