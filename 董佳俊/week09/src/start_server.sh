#!/bin/bash
# 启动 vLLM OpenAI 兼容 server
#
# 教学重点：
#   1. 一条命令把 HuggingFace 模型变成 OpenAI 兼容 API
#   2. 关键启动参数：max-model-len / gpu-memory-utilization / dtype
#   3. 启动后访问 http://localhost:8000/v1/chat/completions 即可调用
#
# 使用方式（在云 GPU 终端内执行）：
#   cd vllm_cloud_deploy/src/
#   bash start_server.sh

set -e

# ── 配置 ─────────────────────────────────────────────────────
# 云 GPU 上直接用 HuggingFace 模型 ID，vLLM 会自动下载
# 若实例已预装模型，改为实际路径（如 /root/autodl-tmp/models/Qwen2-0.5B-Instruct）
MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
SERVED_NAME="qwen2-0.5b"    # 客户端 API 里使用的模型名
PORT=8000
MAX_MODEL_LEN=2048          # 最大上下文长度（0.5B 模型不需要太长）
GPU_MEM_UTIL=0.6            # 占用 60% 显存（给 bench 测试留余地）
DTYPE="float16"

# ── 防止 torch/numpy OpenMP 冲突 ─────────────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================"
echo "  启动 vLLM OpenAI Server"
echo "  模型: $MODEL_PATH"
echo "  对外名称: $SERVED_NAME"
echo "  端口: $PORT"
echo "  max_len: $MAX_MODEL_LEN"
echo "  显存占用: ${GPU_MEM_UTIL} (约 5GB)"
echo "============================================"
echo ""
echo "启动后另开终端测试："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 0.0.0.0
