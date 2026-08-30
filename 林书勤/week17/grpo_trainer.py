"""
GRPO训练器 - Group Relative Policy Optimization
GRPO Trainer for Reinforcement Learning
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import json

from config import GRPOConfig
from math_dataset import MathDataset, format_prompt, parse_model_output, collate_fn


class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(self, config: GRPOConfig):
        """
        初始化GRPO训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载模型和分词器
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # 确保有pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载策略模型（待训练）
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32
        ).to(self.device)
        
        # 加载参考模型（冻结，用于计算KL散度）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32
        ).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        # 加载数据集
        print("Loading datasets...")
        self.train_dataset = MathDataset(
            split=config.dataset_split,
            dataset_name=config.dataset_name
        )
        self.val_dataset = MathDataset(
            split=config.val_split,
            dataset_name=config.dataset_name
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 训练统计
        self.global_step = 0
        self.training_stats = []
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
    
    def generate_responses(
        self,
        questions: List[str],
        num_responses: int
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        为每个问题生成多个回答
        
        Args:
            questions: 问题列表
            num_responses: 每个问题生成的回答数量
            
        Returns:
            (responses, log_probs, ref_log_probs) 元组
        """
        all_responses = []
        all_log_probs = []
        all_ref_log_probs = []
        
        self.policy_model.eval()
        
        for question in questions:
            prompt = format_prompt(question)
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            prompt_length = inputs["input_ids"].shape[1]
            
            responses = []
            log_probs_list = []
            ref_log_probs_list = []
            
            # 生成多个回答
            for _ in range(num_responses):
                with torch.no_grad():
                    # 使用策略模型生成
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    response_ids = outputs.sequences[0][prompt_length:]
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    responses.append(response)
                    
                    # 计算对数概率
                    full_ids = outputs.sequences
                    
                    # 策略模型的对数概率
                    policy_outputs = self.policy_model(full_ids)
                    policy_logits = policy_outputs.logits[:, prompt_length-1:-1, :]
                    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                    
                    # 获取生成token的对数概率
                    response_log_probs = torch.gather(
                        policy_log_probs,
                        2,
                        response_ids.unsqueeze(0).unsqueeze(-1)
                    ).squeeze(-1)
                    
                    log_probs_list.append(response_log_probs)
                    
                    # 参考模型的对数概率
                    ref_outputs = self.ref_model(full_ids)
                    ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    
                    ref_response_log_probs = torch.gather(
                        ref_log_probs,
                        2,
                        response_ids.unsqueeze(0).unsqueeze(-1)
                    ).squeeze(-1)
                    
                    ref_log_probs_list.append(ref_response_log_probs)
            
            all_responses.append(responses)
            all_log_probs.append(log_probs_list)
            all_ref_log_probs.append(ref_log_probs_list)
        
        return all_responses, all_log_probs, all_ref_log_probs
    
    def compute_rewards(
        self,
        responses: List[List[str]],
        ground_truths: List[str]
    ) -> List[List[float]]:
        """
        计算每个回答的奖励
        
        Args:
            responses: 生成的回答 [batch_size, group_size]
            ground_truths: 真实答案
            
        Returns:
            奖励列表 [batch_size, group_size]
        """
        all_rewards = []
        
        for question_responses, gt in zip(responses, ground_truths):
            rewards = []
            for response in question_responses:
                # 解析模型输出
                predicted_answer = parse_model_output(response)
                
                # 检查答案是否正确
                is_correct = MathDataset.check_answer(predicted_answer, gt)
                
                # 分配奖励
                reward = self.config.correct_reward if is_correct else self.config.incorrect_reward
                rewards.append(reward)
            
            all_rewards.append(rewards)
        
        return all_rewards
    
    def compute_grpo_loss(
        self,
        log_probs: List[List[torch.Tensor]],
        ref_log_probs: List[List[torch.Tensor]],
        rewards: List[List[float]]
    ) -> torch.Tensor:
        """
        计算GRPO损失
        
        GRPO的核心思想：
        1. 对每个问题生成多个回答（group）
        2. 基于回答的相对质量（组内比较）来优化策略
        3. 使用KL散度约束避免偏离参考模型太远
        
        Args:
            log_probs: 策略模型的对数概率
            ref_log_probs: 参考模型的对数概率
            rewards: 奖励
            
        Returns:
            损失值
        """
        total_loss = 0.0
        num_groups = len(log_probs)
        
        for group_log_probs, group_ref_log_probs, group_rewards in zip(
            log_probs, ref_log_probs, rewards
        ):
            # 转换为tensor
            group_rewards_tensor = torch.tensor(
                group_rewards,
                dtype=torch.float32,
                device=self.device
            )
            
            # 计算组内相对优势（相对于组平均值）
            reward_mean = group_rewards_tensor.mean()
            reward_std = group_rewards_tensor.std() + 1e-8
            advantages = (group_rewards_tensor - reward_mean) / reward_std
            
            # 计算每个回答的策略损失和KL散度
            for log_prob, ref_log_prob, advantage in zip(
                group_log_probs, group_ref_log_probs, advantages
            ):
                # 平均对数概率
                avg_log_prob = log_prob.mean()
                avg_ref_log_prob = ref_log_prob.mean()
                
                # KL散度 (使用对数概率的差值近似)
                kl_div = avg_log_prob - avg_ref_log_prob
                
                # GRPO策略梯度损失
                policy_loss = -advantage * avg_log_prob
                
                # 总损失：策略损失 + KL惩罚
                loss = policy_loss + self.config.kl_coef * kl_div.abs()
                
                total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / (num_groups * self.config.group_size)
        
        return avg_loss
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            训练统计信息
        """
        self.policy_model.train()
        
        questions = batch["questions"]
        ground_truths = batch["final_answers"]
        
        # 生成回答
        responses, log_probs, ref_log_probs = self.generate_responses(
            questions,
            self.config.group_size
        )
        
        # 计算奖励
        rewards = self.compute_rewards(responses, ground_truths)
        
        # 计算损失
        loss = self.compute_grpo_loss(log_probs, ref_log_probs, rewards)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # 统计信息
        avg_reward = np.mean([np.mean(r) for r in rewards])
        max_reward = np.max([np.max(r) for r in rewards])
        
        stats = {
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "max_reward": max_reward,
        }
        
        return stats
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            评估指标
        """
        self.policy_model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                questions = batch["questions"]
                ground_truths = batch["final_answers"]
                
                # 生成回答（每个问题只生成一个）
                responses, _, _ = self.generate_responses(questions, num_responses=1)
                
                # 检查正确性
                for response_group, gt in zip(responses, ground_truths):
                    response = response_group[0]
                    predicted = parse_model_output(response)
                    is_correct = MathDataset.check_answer(predicted, gt)
                    
                    if is_correct:
                        total_correct += 1
                    total_samples += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": total_correct,
            "total": total_samples
        }
    
    def train(self):
        """执行完整训练"""
        print(f"\n{'='*60}")
        print("Starting GRPO Training")
        print(f"{'='*60}\n")
        
        # 训练前评估
        print("Initial evaluation...")
        initial_metrics = self.evaluate()
        print(f"Initial accuracy: {initial_metrics['accuracy']:.4f}")
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}\n")
            
            epoch_stats = []
            
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # 训练步骤
                stats = self.train_step(batch)
                epoch_stats.append(stats)
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{stats['loss']:.4f}",
                    "reward": f"{stats['avg_reward']:.2f}"
                })
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    avg_stats = {
                        k: np.mean([s[k] for s in epoch_stats[-self.config.logging_steps:]])
                        for k in stats.keys()
                    }
                    self.training_stats.append({
                        "step": self.global_step,
                        **avg_stats
                    })
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # Epoch结束评估
            print(f"\nEvaluating epoch {epoch + 1}...")
            eval_metrics = self.evaluate()
            print(f"Epoch {epoch + 1} accuracy: {eval_metrics['accuracy']:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(f"epoch-{epoch+1}")
        
        # 最终评估
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}\n")
        final_metrics = self.evaluate()
        print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Improvement: {final_metrics['accuracy'] - initial_metrics['accuracy']:.4f}")
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        # 保存训练统计
        self.save_training_stats()
        
        print(f"\nTraining completed! Model saved to: {self.config.output_dir}")
    
    def save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存优化器状态
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            os.path.join(checkpoint_dir, "optimizer.pt")
        )
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def save_training_stats(self):
        """保存训练统计信息"""
        stats_file = os.path.join(self.config.output_dir, "training_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"Training stats saved: {stats_file}")


if __name__ == "__main__":
    # 创建配置
    config = GRPOConfig()
    
    # 创建训练器
    trainer = GRPOTrainer(config)
    
    # 开始训练
    trainer.train()
