# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/7/2 18:13  
@脚本说明 : 

"""
# !/usr/bin/env python3
"""
vLLM 大模型服务一键部署脚本
功能：自动检查环境、安装依赖、下载模型、启动 OpenAI 兼容的 API 服务
"""

import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """检查 Python 版本是否满足要求（>= 3.10）[reference:0]"""
    if sys.version_info < (3, 10):
        logger.error("Python 版本必须 >= 3.10，当前版本: %s", sys.version_info)
        sys.exit(1)
    logger.info("Python 版本检查通过: %s", sys.version)


def check_gpu():
    """检查 NVIDIA GPU 是否可用"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            logger.info("检测到 GPU: %s", gpu_name)
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    logger.warning("未检测到 NVIDIA GPU，将使用 CPU 模式（性能可能较差）")
    return False


def install_vllm():
    """安装 vLLM 及相关依赖[reference:1][reference:2]"""
    logger.info("正在安装 vLLM 和 transformers...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "vllm", "transformers", "-q"],
            check=True
        )
        logger.info("vLLM 安装完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("vLLM 安装失败: %s", e)
        return False


def download_model(model_name, cache_dir=None):
    """
    下载模型到本地缓存（使用 huggingface_hub）
    如果模型已存在则跳过
    """
    try:
        from huggingface_hub import snapshot_download
        logger.info("正在检查/下载模型: %s", model_name)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            ignore_patterns=["*.safetensors", "*.bin"]  # 先跳过大文件快速检查
        )
        logger.info("模型已就绪: %s", local_path)
        return local_path
    except ImportError:
        logger.info("huggingface_hub 未安装，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"],
            check=True
        )
        return download_model(model_name, cache_dir)
    except Exception as e:
        logger.warning("模型下载失败（可能已存在）: %s", e)
        return model_name


def start_vllm_server(model_name, host="0.0.0.0", port=8000, **kwargs):
    """
    启动 vLLM HTTP 服务器（兼容 OpenAI API）[reference:3]

    参数说明：
        --tensor-parallel-size: 张量并行的 GPU 数量[reference:4]
        --gpu-memory-utilization: GPU 显存利用率 (0.0-1.0)[reference:5]
        --max-model-len: 模型最大上下文长度[reference:6]
        --dtype: 计算精度 (auto/float16/bfloat16/float32)[reference:7]
        --trust-remote-code: 允许执行自定义模型代码[reference:8]
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
    ]

    # 添加额外参数
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    logger.info("启动 vLLM 服务器...")
    logger.info("命令: %s", " ".join(cmd))
    logger.info("服务地址: http://%s:%s", host if host != "0.0.0.0" else "localhost", port)
    logger.info("API 文档: http://%s:%s/docs", host if host != "0.0.0.0" else "localhost", port)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("服务已停止")
    except subprocess.CalledProcessError as e:
        logger.error("服务启动失败: %s", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="vLLM 大模型服务一键部署脚本")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="模型名称（HuggingFace ID 或本地路径）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务绑定的主机地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="张量并行的 GPU 数量（默认: 1）[reference:9]"
    )
    parser.add_argument(
        "--gpu-memory-utilization", "-mem",
        type=float,
        default=0.9,
        help="GPU 显存利用率（默认: 0.9）[reference:10]"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="模型最大上下文长度[reference:11]"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="计算精度（默认: auto）[reference:12]"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="模型缓存目录"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过模型下载（假设模型已存在）"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="跳过 vLLM 安装"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行自定义模型代码[reference:13]"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要执行的命令，不实际运行[reference:14]"
    )

    args = parser.parse_args()

    # 1. 环境检查
    logger.info("=" * 60)
    logger.info("vLLM 大模型服务部署脚本")
    logger.info("=" * 60)
    check_python_version()
    has_gpu = check_gpu()

    if not has_gpu:
        logger.warning("CPU 模式下建议使用较小的模型或启用 --dtype float32")

    # 2. 安装依赖
    if not args.skip_install:
        if not install_vllm():
            logger.error("依赖安装失败，请手动安装: pip install vllm transformers")
            sys.exit(1)
    else:
        logger.info("跳过依赖安装")

    # 3. 下载模型
    model_path = args.model
    if not args.skip_download and not os.path.exists(args.model):
        model_path = download_model(args.model, args.cache_dir)
    else:
        logger.info("使用已存在的模型: %s", args.model)

    # 4. 构建启动参数
    server_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    # 移除 None 值
    server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}

    # 5. 启动服务
    if args.dry_run:
        logger.info("=== DRY RUN 模式 ===")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", args.host,
            "--port", str(args.port),
        ]
        for key, value in server_kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        logger.info("将执行: %s", " ".join(cmd))
        return

    start_vllm_server(
        model_name=model_path,
        host=args.host,
        port=args.port,
        **server_kwargs
    )


if __name__ == "__main__":
    main()