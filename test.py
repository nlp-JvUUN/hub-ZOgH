import torch

# 检查 MPS 是否可用（我的本地硬件环境：mac电脑M芯片，支持GPU运算）
print(f"MPS available: {torch.backends.mps.is_available()}")
# 设置PyTorch的计算设备（Device）
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
