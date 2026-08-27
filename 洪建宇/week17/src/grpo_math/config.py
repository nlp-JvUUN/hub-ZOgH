"""GRPO 训练配置。集中管理所有超参与路径，便于命令行覆盖。"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GRPOConfig:
    # ---- 模型 ----
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    ref_model_name_or_path: Optional[str] = None  # None 时复用 policy 路径
    output_dir: str = "./outputs"

    # ---- 数据 ----
    dataset_name: str = "gsm8k"          # "gsm8k" | "math"
    dataset_split: str = "train"
    num_train_samples: int = 1000       # -1 表示使用全部
    max_prompt_length: int = 512
    max_new_tokens: int = 512

    # ---- GRPO ----
    num_generations: int = 8            # G：每个 prompt 的组采样数
    temperature: float = 1.0
    top_p: float = 1.0
    clip_eps: float = 0.2               # PPO/GRPO 截断比例
    beta: float = 0.04                  # 相对参考模型的 KL 系数
    advantage_eps: float = 1e-8        # 组内归一化防除零

    # ---- 训练 ----
    learning_rate: float = 1e-6
    num_epochs: int = 1
    prompt_batch_size: int = 4          # 每步处理多少个 prompt（展开后 = *G）
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    # ---- 奖励权重 ----
    reward_correctness_weight: float = 1.0
    reward_format_weight: float = 0.2

    # ---- 采样 dtype ----
    bf16: bool = True

    # ---- 显存控制 ----
    logps_chunk_size: int = 4          # 计算 per-token logp 时按 batch 维分块，避免 OOM
    gradient_checkpointing: bool = True

    def to_dict(self):
        return asdict(self)
