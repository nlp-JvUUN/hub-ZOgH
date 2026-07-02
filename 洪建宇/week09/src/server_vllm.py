"""
使用 vLLM 部署大模型服务（支持 OpenAI 兼容 API）

vLLM 核心优势：
  1. 高吞吐量：PagedAttention 机制，显存利用率提升 2~3 倍
  2. 低延迟：连续 batching，请求无需等待上一个完成
  3. OpenAI 兼容：/v1/completions 和 /v1/chat/completions 接口
  4. 动态批处理：自动合并多个请求到同一批次

使用方式：
  # ── 启动基础模型服务 ──────────────────────────────────────────────────────
  python server_vllm.py --model_path ../../pretrain_models/Qwen2-0.5B-Instruct

  # ── 启动 LoRA 微调后的服务（自动合并 adapter）─────────────────────────────
  python server_vllm.py --model_path ../../pretrain_models/Qwen2-0.5B-Instruct \
                        --lora_adapter ../outputs/sft_adapter

  # ── 启动全量微调后的服务 ──────────────────────────────────────────────────
  python server_vllm.py --model_path ../outputs/sft_full_ckpt

  # ── 常用参数 ──────────────────────────────────────────────────────────────
  python server_vllm.py --model_path ... --port 8000 --tensor-parallel-size 1
  python server_vllm.py --model_path ... --max-num-batched-tokens 2048

验证服务：
  # 方式1：curl 测试
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen2-0.5B-Instruct",
      "messages": [{"role": "user", "content": "新闻标题：苹果发布新款iPhone\n类别："}]
    }'

  # 方式2：使用 OpenAI Python SDK
  pip install openai
  python -c "
    from openai import OpenAI
    client = OpenAI(base_url='http://localhost:8000/v1', api_key='sk-xxx')
    response = client.chat.completions.create(
        model='Qwen2-0.5B-Instruct',
        messages=[{'role': 'user', 'content': '新闻标题：苹果发布新款iPhone\n类别：'}]
    )
    print(response.choices[0].message.content)
  "

依赖：
  pip install vllm>=0.6.0

注意：
  - vLLM 需要 NVIDIA GPU 和 CUDA 环境
  - Windows 环境建议使用 WSL2 运行
  - 首次启动会加载模型到显存，耗时较长（约 1~2 分钟）
"""

import os
import argparse
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM 大模型服务启动脚本")

    parser.add_argument("--model_path", type=str, required=True,
                        help="基础模型路径或全量微调模型路径")

    parser.add_argument("--lora_adapter", type=str, default=None,
                        help="LoRA adapter 路径；指定后自动合并到基础模型")

    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务监听地址，默认 0.0.0.0（允许外部访问）")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口，默认 8000")

    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="张量并行度，多卡时设置 >1；单卡设为 1")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048,
                        help="批处理最大 token 数，越大吞吐量越高，但显存占用越大")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="模型最大序列长度，不指定则使用模型默认值")
    parser.add_argument("--quantization", type=str, default=None,
                        help="量化方式：awq/gptq/squeezellm，需模型支持")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU 显存利用率，默认 0.9（90%）")

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="温度参数，0.0 为 greedy decoding（确定性输出）")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="最大生成 token 数")
    parser.add_argument("--stop", type=str, nargs="*", default=["<|im_end|>", "\n"],
                        help="停止词，遇到这些词就停止生成")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from vllm import LLM, EngineArgs
    except ImportError as e:
        print(f"[错误] 无法导入 vLLM: {e}")
        print("请安装 vLLM: pip install vllm>=0.6.0")
        print("注意：vLLM 需要 NVIDIA GPU 和 CUDA 环境")
        print("Windows 用户建议使用 WSL2")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[错误] 模型路径不存在: {model_path}")
        sys.exit(1)

    lora_path = None
    if args.lora_adapter:
        lora_path = Path(args.lora_adapter)
        if not lora_path.exists():
            print(f"[错误] LoRA adapter 路径不存在: {lora_path}")
            sys.exit(1)
        print(f"检测到 LoRA adapter，将合并到基础模型")

    engine_args = EngineArgs(
        model=str(model_path.resolve()),
        tokenizer=str(model_path.resolve()),
        lora_path=str(lora_path.resolve()) if lora_path else None,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )

    print(f"正在加载模型: {model_path}")
    if lora_path:
        print(f"合并 LoRA adapter: {lora_path}")

    try:
        llm = LLM(engine_args=engine_args)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        print("常见原因：")
        print("  1. CUDA 环境未配置")
        print("  2. GPU 显存不足")
        print("  3. 模型文件损坏或不完整")
        print("  4. Windows 环境建议使用 WSL2")
        sys.exit(1)

    print("模型加载完成\n")

    print("=" * 60)
    print("vLLM 服务已启动")
    print("=" * 60)
    print(f"  模型: {model_path.name}")
    print(f"  LoRA adapter: {lora_path.name if lora_path else '无'}")
    print(f"  服务地址: http://{args.host}:{args.port}")
    print(f"  张量并行: {args.tensor_parallel_size}")
    print(f"  最大批处理 token: {args.max_num_batched_tokens}")
    print(f"  默认温度: {args.temperature}")
    print(f"  默认最大生成: {args.max_new_tokens}")
    print("-" * 60)
    print("可用 API 端点:")
    print(f"  GET   http://{args.host}:{args.port}/v1/models")
    print(f"  POST  http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  POST  http://{args.host}:{args.port}/v1/completions")
    print("-" * 60)
    print("测试命令:")
    print(f"  curl http://{args.host}:{args.port}/v1/chat/completions \\\\")
    print(f"    -H 'Content-Type: application/json' \\\\")
    print(f"    -d '{{\"model\": \"{model_path.name}\", \"messages\": [{{\"role\": \"user\", \"content\": \"你好\"}}]}}'")
    print("=" * 60)

    try:
        from vllm.entrypoints.openai.api_server import OpenAIServer
        server = OpenAIServer(llm=llm, engine_args=engine_args)
    except ImportError:
        try:
            from vllm.entrypoints.openai.api_server import create_app
            from vllm.engine.arg_utils import AsyncEngineArgs
            async_engine_args = AsyncEngineArgs(
                model=str(model_path.resolve()),
                tokenizer=str(model_path.resolve()),
                lora_path=str(lora_path.resolve()) if lora_path else None,
                tensor_parallel_size=args.tensor_parallel_size,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_model_len=args.max_model_len,
                quantization=args.quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="auto",
            )
            server = create_app(async_engine_args)
        except Exception as e2:
            print(f"[错误] 创建 OpenAI API 服务失败: {e2}")
            print("尝试使用 vllm serve 命令启动:")
            print(f"  vllm serve --model {model_path} --host {args.host} --port {args.port}")
            if lora_path:
                print(f"  vllm serve --model {model_path} --lora-path {lora_path} --host {args.host} --port {args.port}")
            sys.exit(1)

    try:
        import uvicorn
        uvicorn.run(
            server.app if hasattr(server, 'app') else server,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except Exception as e:
        print(f"[错误] 启动服务失败: {e}")
        print("尝试直接使用 vllm serve 命令:")
        cmd = f"vllm serve --model {model_path} --host {args.host} --port {args.port}"
        if lora_path:
            cmd += f" --lora-path {lora_path}"
        print(f"  {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
