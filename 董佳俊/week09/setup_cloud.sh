#!/bin/bash
# setup_cloud.sh — 云 GPU 实例环境一键配置
#
# 使用方式（在云 GPU 终端内执行）：
#   bash setup_cloud.sh
#
# 适用平台：AutoDL / 矩池云 / 任意 Ubuntu 22.04 + NVIDIA GPU
# 耗时：约 5 分钟（取决于网速）
# 磁盘占用：模型约 1GB + 依赖约 3GB

set -e

echo "============================================"
echo "  vLLM 云 GPU 环境配置"
echo "============================================"
echo ""

# ── 1. 检查 GPU ──────────────────────────────────────────────────
echo "[1/5] 检查 GPU 环境..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ nvidia-smi 未找到，请确认实例已分配 GPU"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# ── 2. 配置 pip 镜像（国内加速）───────────────────────────────────
echo "[2/5] 配置 pip 清华源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "  ✓ pip 源已切换为清华镜像"
echo ""

# ── 3. 安装 Python 依赖 ──────────────────────────────────────────
echo "[3/5] 安装 Python 依赖（约 2-3 分钟）..."
pip install --upgrade pip -q
pip install vllm==0.9.2
pip install torch==2.7.0
pip install transformers==4.52.4
pip install accelerate>=1.0.0
pip install "openai>=1.40.0"
pip install "jsonschema>=4.20.0"
pip install "matplotlib>=3.7.0"
pip install "numpy>=1.26.0"
pip install "httpx>=0.25.0"
echo "  ✓ 依赖安装完成"
echo ""

# ── 4. 验证环境 ──────────────────────────────────────────────────
echo "[4/5] 验证环境..."
python -c "
import vllm, torch
print(f'  vLLM:       {vllm.__version__}')
print(f'  CUDA 可用:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  显存:       {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
"
echo ""

# ── 5. 显示后续步骤 ──────────────────────────────────────────────
echo "[5/5] 环境配置完成！"
echo ""
echo "============================================"
echo "  接下来请执行："
echo "============================================"
echo "  # 进入源码目录"
echo "  cd vllm_cloud_deploy/src/"
echo ""
echo "  # 第一步：启动 vLLM server（阻塞，新开终端）"
echo "  bash start_server.sh"
echo ""
echo "  # 第二步：另开终端，跑约束解码 demo（5选1）"
echo "  python demo_guided_choice.py      # 枚举约束"
echo "  python demo_guided_regex.py       # 正则约束"
echo "  python demo_guided_json.py        # JSON Schema 基础"
echo "  python demo_response_format.py    # OpenAI 标准 json_object"
echo "  python demo_function_call.py      # ★ 核心：双工具×50×3"
echo ""
echo "  # 第三步：吞吐 benchmark（先停 server！）"
echo "  fuser -k 8000/tcp"
echo "  python bench_throughput.py        # 3 路对比"
echo "============================================"
