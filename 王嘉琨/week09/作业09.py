from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 配置模型和推理参数
sampling_params = SamplingParams(
    temperature=0.7,  # 随机性（0为确定性）
    max_tokens=200    # 最大生成token数
)
llm = LLM(
    model="lmsys/vicuna-7b-v1.5",  # 模型路径
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9  # 允许使用的GPU显存比例（0.9即90%）
)

# 定义请求格式
class Request(BaseModel):
    prompt: str

# 定义生成接口
@app.post("/generate")
async def generate(request: Request):
    outputs = llm.generate(request.prompt, sampling_params)
    return {"response": outputs[0].outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "请介绍一下人工智能的发展历程。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=200)
end = time.time()
print(f"原生transformers耗时：{end - start:.2f}秒")
