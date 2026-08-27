"""GRPO 数学题强化学习训练框架。

模块组成：
  config.py        超参数配置
  data.py          数据格式（GSM8K/MATH 加载与 chat prompt）
  rewards.py       可验证奖励函数（正确性 + 格式）
  grpo_trainer.py  组采样 → 优势计算 → 策略更新 → 训练循环
  train.py         命令行入口
"""
from .config import GRPOConfig
from .grpo_trainer import GRPOTrainer

__all__ = ["GRPOConfig", "GRPOTrainer"]
