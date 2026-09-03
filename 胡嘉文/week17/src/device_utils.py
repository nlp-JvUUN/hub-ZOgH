"""
跨平台设备 / dtype / 模型路径工具：CUDA (Windows) 与 Apple Silicon (MPS) / CPU 自动适配。

Mac 与 Windows 的关键差异：
  1. 设备：无 CUDA → MPS（Metal）或 CPU。用 torch.cuda.is_available() /
     torch.backends.mps.is_available() 自动选择，也可用环境变量 GRPO_DEVICE 强制。
  2. dtype：CUDA 用 bfloat16（原项目的防溢出方案，见 USAGE_GUIDE Q3）；
     MPS / CPU 用 float32——MPS 对 bf16 支持不完整，而 fp16 会让 AdamW 的
     eps=1e-8 溢出为 0 → 0/0=NaN 一步训废模型（原项目已实测）。fp32 最稳。
  3. 模型路径：原项目写死 Windows 的 D:\\badou\\...，这里按
     命令行参数 → 环境变量 MODEL_PATH → 本地 pretrain_models 目录 的顺序解析，
     不下载模型（本机已有落盘模型）。
"""
import os
from pathlib import Path

import torch

# 本地基座模型：Mac 上的模型已落盘（约 1GB），无需下载。
# 如需换路径，设置环境变量 MODEL_PATH，或用命令行 --model 覆盖。
DEFAULT_LOCAL = Path("/Users/hjw/文档/八斗学院AI/pretrain_models/Qwen2-0.5B-Instruct")


def get_device() -> str:
    """自动选择设备：CUDA > MPS > CPU。可用环境变量 GRPO_DEVICE 手动覆盖。"""
    forced = os.environ.get("GRPO_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def get_dtype():
    """dtype 选择：CUDA 用 bfloat16（防 fp16 溢出）；MPS/CPU 用 float32。"""
    if get_device() == "cuda":
        return torch.bfloat16
    return torch.float32


def get_attn_implementation():
    """注意力实现。MPS 上 torch 2.6 的 sdpa + GQA（Qwen2-0.5B 14 头 Q / 4 头 KV）
    会报 mps.matmul 维度不匹配崩溃（已实测），必须退回 eager；
    CUDA/CPU 返回 None，让 transformers 用默认实现（CUDA 下 sdpa，更快）。"""
    if get_device() == "mps":
        return "eager"
    return None


def device_summary() -> str:
    """人类可读的设备描述，用于日志打印。"""
    device = get_device()
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def peak_memory_gb() -> float:
    """GPU 峰值显存（GB）。MPS/CPU 无此统计，返回 0。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def empty_device_cache() -> None:
    """释放设备缓存：CUDA 有效；MPS/CPU 空操作。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_path(model_arg=None) -> str:
    """解析模型路径。优先级：命令行参数 > 环境变量 MODEL_PATH > 本地 pretrain_models 目录。"""
    explicit = model_arg or os.environ.get("MODEL_PATH")
    if explicit:
        return str(Path(explicit))
    return str(DEFAULT_LOCAL)


def _has_weights(model_dir: Path) -> bool:
    """检查目录里是否有真正的模型权重（safetensors / bin，排除 LoRA adapter）。"""
    if not model_dir.exists():
        return False
    for p in model_dir.iterdir():
        if p.suffix == ".safetensors" and p.name != "adapter_model.safetensors":
            return True
        if p.name == "pytorch_model.bin":
            return True
    return False


def ensure_model(model_path) -> str:
    """校验模型路径存在，返回规范化路径。不做下载。

    兼容两类目录：
      - 完整模型：含 config.json + model.safetensors
      - LoRA adapter：只含 adapter_config.json + adapter_model.safetensors
        （不含基座 config，加载时由 probe 脚本先挂基座）
    """
    p = Path(model_path)
    is_full = (p / "config.json").exists()
    is_adapter = (p / "adapter_config.json").exists()
    if is_full or is_adapter:
        if is_full and not _has_weights(p):
            print(
                f"[警告] {p} 只有 config/tokenizer，缺少模型权重文件"
                f"（*.safetensors / pytorch_model.bin）。"
                f"可能是残缺的 checkpoint，加载很可能失败。"
            )
        return str(p)
    raise FileNotFoundError(
        f"模型路径不存在: {p}\n"
        f"请设置环境变量 MODEL_PATH 指向含 config.json + model.safetensors 的模型目录。"
    )
