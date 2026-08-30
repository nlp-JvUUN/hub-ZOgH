"""
训练主程序
Main Training Script
"""

import argparse
import os
import torch

from config import GRPOConfig
from grpo_trainer import GRPOTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GRPO训练用于数学问题求解")
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型名称"
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO组大小")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL散度系数")
    
    # 数据参数
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:1000]",
        help="训练集切分"
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="test[:100]",
        help="验证集切分"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo_math_model",
        help="输出目录"
    )
    
    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = GRPOConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        dataset_split=args.dataset_split,
        val_split=args.val_split,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    # 打印配置
    print("\n" + "="*60)
    print("GRPO Training Configuration")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Group size: {config.group_size}")
    print(f"KL coefficient: {config.kl_coef}")
    print(f"Dataset split: {config.dataset_split}")
    print(f"Output directory: {config.output_dir}")
    print("="*60 + "\n")
    
    # 创建训练器
    trainer = GRPOTrainer(config)
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
        print("Checkpoint saved. You can resume training later.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
