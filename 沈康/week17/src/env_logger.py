"""
环境信息收集与日志打印（在训练/评估开始前运行）

用途：
  1. 启动时在控制台打印环境摘要，一眼确认 GPU / 依赖 / CUDA 是否符合预期，
     避免在服务器上"跑完了才发现装的是 CPU 版 torch"这类浪费
  2. 把完整环境快照追加保存到 outputs/env_info.json（多次运行累积为列表），
     该文件随 outputs/ 一起下载回本地，作为大模型分析报告的输入材料

使用方式（各脚本 main() 开头调用一次）：
  from env_logger import log_env
  log_env(OUT_DIR, tag="train", extra={"max_steps": 200})

自检：直接运行本文件可独立验证收集功能（快照写到临时目录，不污染 outputs）：
  python src/env_logger.py
"""
import json
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path

# 需要记录版本的关键依赖（与 requirements.txt 对应）
PKGS = ["transformers", "trl", "peft", "accelerate", "datasets", "numpy"]


def _pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "未安装"


def collect_env_info(tag: str = "", extra: dict | None = None) -> dict:
    """收集环境快照。只依赖标准库 + torch，失败项记录原因而不是中断脚本。"""
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 1024**3, 2)
            info["cuda_version"] = torch.version.cuda
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
        else:
            info["gpu"] = None
            info["cuda_version"] = torch.version.cuda  # 编译时的 CUDA 版本（+cpu 则为 None）
    except Exception as e:
        info["torch"] = f"导入失败: {e}"

    for pkg in PKGS:
        info[f"pkg_{pkg}"] = _pkg_version(pkg)

    if extra:
        info.update(extra)
    return info


def log_env(out_dir, tag: str = "", extra: dict | None = None) -> dict:
    """打印环境摘要到控制台，并把快照追加到 out_dir/env_info.json。"""
    info = collect_env_info(tag, extra)

    gpu_txt = info.get("gpu") or "未检测到 CUDA GPU"
    if info.get("gpu"):
        gpu_txt += f"（{info.get('gpu_mem_gb')} GB, bf16={info.get('bf16_supported')}）"
    print("=" * 68)
    print(f"[环境检查] {info['timestamp']}  tag={tag or '-'}")
    print(f"  主机: {info['hostname']}   系统: {info['os']}   CPU 核心: {info['cpu_count']}")
    print(f"  Python: {info['python']}   torch: {info.get('torch')}")
    print(f"  CUDA available: {info.get('cuda_available')}   CUDA 版本: {info.get('cuda_version')}")
    print(f"  GPU: {gpu_txt}")
    print("  依赖: " + "  ".join(f"{p}={info.get(f'pkg_{p}')}" for p in PKGS))
    print("=" * 68)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    snap_file = out_path / "env_info.json"
    snapshots = []
    if snap_file.exists():
        try:
            snapshots = json.loads(snap_file.read_text(encoding="utf-8"))
            if not isinstance(snapshots, list):
                snapshots = [snapshots]
        except Exception:
            snapshots = []
    snapshots.append(info)
    snap_file.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    # 自检：写到系统临时目录，避免污染项目的 outputs/
    log_env(Path(tempfile.gettempdir()) / "env_logger_selftest", tag="selftest")
    print("自检通过。快照已写入临时目录（不影响项目 outputs/）。")

