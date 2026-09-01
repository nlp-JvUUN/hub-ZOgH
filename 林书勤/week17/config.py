"""
配置文件 - GRPO强化学习数学问题求解
Configuration for GRPO Reinforcement Learning on Math Problems
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # 基础模型
    
    # 训练配置
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_length: int = 512
    
    # GRPO特定参数
    group_size: int = 4  # 每个问题生成的答案数量
    kl_coef: float = 0.1  # KL散度系数
    clip_range: float = 0.2  # PPO裁剪范围
    gamma: float = 1.0  # 折扣因子
    
    # 数据配置
    dataset_name: str = "gsm8k"  # 数学数据集
    dataset_split: str = "train[:1000]"  # 使用部分数据进行快速实验
    val_split: str = "test[:100]"
    
    # 奖励配置
    correct_reward: float = 1.0  # 正确答案奖励
    incorrect_reward: float = -0.5  # 错误答案惩罚
    
    # 输出配置
    output_dir: str = "./grpo_math_model"
    save_steps: int = 100
    logging_steps: int = 10
    
    # 设备配置
    device: str = "cuda"  # 或 "cpu"
    mixed_precision: bool = True  # 混合精度训练
    
    # 分布式配置（可选）
    use_distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    
    # 其他
    seed: int = 42
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2
