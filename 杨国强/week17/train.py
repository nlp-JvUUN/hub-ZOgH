"""Main training script for GRPO-based math problem training."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.grpo import GRPO, GRPOConfig
from core.policy import Policy, create_policy, MockPolicy
from core.buffer import ExperienceBuffer
from envs.math_env import MathEnvironment, ProblemCategory, ProblemDifficulty
from rewards.reward_model import RewardModel, RewardConfig
from utils.config import TrainingConfig, load_config, save_config
from utils.logger import TrainingLogger, setup_logging


class GRPOTrainer:
    """Main trainer class for GRPO math training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self._init_policy()
        self._init_grpo()
        self._init_environment()
        self._init_reward_model()
        self._init_buffer()
        self._init_logger()
        
        # Training state
        self.global_step = 0
        self.iteration = 0
        self.best_reward = -float('inf')
        
    def _init_policy(self):
        """Initialize the policy model."""
        if self.config.policy_type == 'mock':
            self.policy = MockPolicy(model_name=self.config.model_name)
        else:
            self.policy = create_policy(
                self.config.policy_type,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        print(f"Initialized {self.config.policy_type} policy with model: {self.config.model_name}")
    
    def _init_grpo(self):
        """Initialize GRPO algorithm."""
        grpo_config = GRPOConfig(
            learning_rate=self.config.learning_rate,
            group_size=self.config.group_size,
            kl_coef=self.config.kl_coef,
            clip_ratio=self.config.clip_ratio,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            entropy_coef=self.config.entropy_coef,
            device=self.device
        )
        self.grpo = GRPO(grpo_config)
        self.optimizer = torch.optim.Adam(
            self.policy.model.parameters() if hasattr(self.policy, 'model') else [torch.tensor(0)],
            lr=self.config.learning_rate
        )
        print(f"Initialized GRPO with group_size={self.config.group_size}, lr={self.config.learning_rate}")
    
    def _init_environment(self):
        """Initialize the math problem environment."""
        categories = [ProblemCategory(c) for c in self.config.problem_categories]
        self.env = MathEnvironment(
            categories=categories,
            difficulty=ProblemDifficulty(self.config.difficulty),
            seed=self.config.seed
        )
        print(f"Initialized environment with categories: {self.config.problem_categories}")
    
    def _init_reward_model(self):
        """Initialize the reward model."""
        reward_config = RewardConfig(
            accuracy_weight=1.0,
            format_weight=0.1,
            length_penalty_weight=-0.01,
            partial_credit_weight=0.5,
            require_step_by_step=self.config.require_step_by_step,
            require_final_answer=True
        )
        self.reward_model = RewardModel(reward_config)
        print("Initialized reward model")
    
    def _init_buffer(self):
        """Initialize experience buffer."""
        self.buffer = ExperienceBuffer(capacity=self.config.buffer_capacity)
        print(f"Initialized experience buffer with capacity {self.config.buffer_capacity}")
    
    def _init_logger(self):
        """Initialize logging."""
        self.logger = TrainingLogger(
            log_dir=self.config.log_dir,
            experiment_name=self.config.experiment_name
        )
        setup_logging(self.config.log_level)
        print(f"Initialized logger, saving to: {self.config.log_dir}")
    
    def generate_responses(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of response dicts with 'response', 'log_prob', etc.
        """
        # Generate with some temperature variation for diversity
        responses = self.policy.generate(
            prompts,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return responses
    
    def collect_experiences(self, n_problems: int) -> Dict[str, Any]:
        """Collect experiences by generating and evaluating responses.
        
        Args:
            n_problems: Number of problems to sample
            
        Returns:
            Dictionary with collected experiences
        """
        # Sample problems
        problems = self.env.sample_problems(n_problems)
        prompts = [self.env.generate_prompt(p) for p in problems]
        
        all_responses = []
        all_rewards = []
        all_log_probs = []
        
        # For each problem, generate G responses (group size)
        for i, (problem, prompt) in enumerate(zip(problems, prompts)):
            # Generate group of responses
            group_responses = self.generate_responses([prompt] * self.config.group_size)
            
            # Compute rewards for each response
            for resp_data in group_responses:
                response_text = resp_data['response']
                log_prob = resp_data.get('log_prob', 0.0)
                
                # Check answer
                is_correct, _ = self.env.check_answer(response_text, problem)
                
                # Compute reward
                reward, _ = self.reward_model.compute_reward(
                    response_text, problem, check_answer_fn=self.env.check_answer
                )
                
                all_responses.append({
                    'prompt': prompt,
                    'response': response_text,
                    'problem': problem,
                    'is_correct': is_correct
                })
                all_rewards.append(reward)
                all_log_probs.append(log_prob)
                
                # Add to buffer
                self.buffer.add(prompt, response_text, reward, log_prob)
        
        return {
            'responses': all_responses,
            'rewards': all_rewards,
            'log_probs': all_log_probs,
            'problems': problems,
            'prompts': prompts
        }
    
    def train_step(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            experiences: Collected experiences
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        # Get batch tensors from buffer
        batch_data = self.buffer.get_group_tensors(device=self.device)
        
        # Compute new log probs - need to match group structure
        # batch_data['log_probs_old'] has shape (n_groups, group_size)
        n_groups = batch_data['rewards'].shape[0]
        group_size = batch_data['rewards'].shape[1]
        
        # Create log_probs_new with same shape as log_probs_old
        # In a real implementation, this would be computed from the model
        # For mock, we add some noise to simulate learning
        log_probs_new = batch_data['log_probs_old'].clone().detach()
        # Add small perturbation to simulate policy update
        log_probs_new = log_probs_new + torch.randn_like(log_probs_new) * 0.1
        # Enable gradients for the mock training step
        log_probs_new.requires_grad_(True)
        
        # Also ensure log_probs_old has grad if needed for the update
        if not batch_data['log_probs_old'].requires_grad:
            batch_data['log_probs_old'] = batch_data['log_probs_old'].detach().requires_grad_(True)
        
        # Add to batch data
        batch_data['log_probs_new'] = log_probs_new
        
        # Update for multiple epochs
        for epoch in range(self.config.n_epochs):
            epoch_metrics = self.grpo.update(
                model=self.policy,
                optimizer=self.optimizer,
                data=batch_data,
                epoch=epoch
            )
            
            for key, value in epoch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics
        for key in metrics:
            if isinstance(metrics[key], list) and len(metrics[key]) > 0:
                metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def evaluate(self, n_problems: int = 100) -> Dict[str, float]:
        """Evaluate the current policy.
        
        Args:
            n_problems: Number of problems to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating on {n_problems} problems...")
        
        problems = self.env.sample_problems(n_problems)
        correct = 0
        total_reward = 0.0
        all_components = []
        
        for problem in tqdm(problems, desc="Evaluating"):
            prompt = self.env.generate_prompt(problem)
            
            # Generate single response (greedy)
            responses = self.policy.generate(
                [prompt],
                temperature=0.0,  # Greedy for evaluation
                max_tokens=self.config.max_tokens
            )
            
            response_text = responses[0]['response']
            
            # Check and reward
            is_correct, _ = self.env.check_answer(response_text, problem)
            reward, components = self.reward_model.compute_reward(
                response_text, problem, check_answer_fn=self.env.check_answer
            )
            
            if is_correct:
                correct += 1
            total_reward += reward
            all_components.append(components)
        
        accuracy = correct / n_problems
        mean_reward = total_reward / n_problems
        
        eval_metrics = {
            'accuracy': accuracy,
            'mean_reward': mean_reward,
            'correct': correct,
            'total': n_problems
        }
        
        # Add reward statistics
        reward_stats = self.reward_model.get_statistics(
            [c['accuracy'] for c in all_components],  # Using accuracy as reward proxy
            all_components
        )
        eval_metrics.update(reward_stats)
        
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{n_problems})")
        print(f"  Mean Reward: {mean_reward:.4f}")
        
        return eval_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'iteration': self.iteration,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'config': self.config.to_dict(),
            'buffer_stats': self.buffer.get_statistics(),
            # Policy state would be saved here for non-mock policies
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.iteration = checkpoint['iteration']
        self.global_step = checkpoint['global_step']
        self.best_reward = checkpoint['best_reward']
        
        print(f"Checkpoint loaded from: {path}")
        print(f"  Iteration: {self.iteration}, Best reward: {self.best_reward:.4f}")
    
    def train(self, n_iterations: int, eval_every: int = 5, save_every: int = 10):
        """Main training loop.
        
        Args:
            n_iterations: Total number of training iterations
            eval_every: Evaluate every N iterations
            save_every: Save checkpoint every N iterations
        """
        print("\n" + "="*60)
        print("Starting GRPO Training")
        print("="*60)
        print(f"Configuration:")
        print(f"  Iterations: {n_iterations}")
        print(f"  Problems per iteration: {self.config.batch_size}")
        print(f"  Group size: {self.config.group_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  KL coefficient: {self.config.kl_coef}")
        print(f"  Eval every: {eval_every}")
        print(f"  Save every: {save_every}")
        print("="*60 + "\n")
        
        for iteration in range(1, n_iterations + 1):
            self.iteration = iteration
            
            print(f"\n{'='*20} Iteration {iteration}/{n_iterations} {'='*20}")
            
            # Collect experiences
            print("Collecting experiences...")
            experiences = self.collect_experiences(self.config.batch_size)
            
            # Log collection stats
            mean_reward = np.mean(experiences['rewards'])
            accuracy = np.mean([int(e['is_correct']) for e in experiences['responses']])
            print(f"  Mean reward: {mean_reward:.4f}")
            print(f"  Accuracy: {accuracy:.2%}")
            
            # Training step
            print("Training...")
            train_metrics = self.train_step(experiences)
            
            # Log metrics
            self.logger.log_metrics({
                'train/mean_reward': mean_reward,
                'train/accuracy': accuracy,
                'train/policy_loss': train_metrics.get('policy_loss', 0),
                'train/kl_div': train_metrics.get('kl_div', 0),
                'train/total_loss': train_metrics.get('total_loss', 0),
            }, step=self.global_step)
            
            self.global_step += 1
            
            # Evaluation
            if iteration % eval_every == 0:
                eval_metrics = self.evaluate(n_problems=50)
                
                self.logger.log_metrics({
                    'eval/accuracy': eval_metrics['accuracy'],
                    'eval/mean_reward': eval_metrics['mean_reward'],
                }, step=self.global_step)
                
                # Track best
                if eval_metrics['mean_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['mean_reward']
                    print(f"  New best reward: {self.best_reward:.4f}")
            
            # Save checkpoint
            if iteration % save_every == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_iter_{iteration}.json"
                )
                self.save_checkpoint(checkpoint_path)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best reward achieved: {self.best_reward:.4f}")
        print("="*60)
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_metrics = self.evaluate(n_problems=200)
        
        return final_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GRPO for math problems")
    
    # Training
    parser.add_argument("--n_iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (problems per iteration)")
    parser.add_argument("--group_size", type=int, default=8,
                        help="Group size for GRPO")
    parser.add_argument("--n_epochs", type=int, default=4,
                        help="Number of training epochs per batch")
    
    # Model
    parser.add_argument("--policy_type", type=str, default="mock",
                        choices=["mock", "openai", "anthropic"],
                        help="Policy type")
    parser.add_argument("--model_name", type=str, default="gpt-4",
                        help="Model name for API policies")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    
    # GRPO
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL divergence coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    
    # Environment
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "expert"],
                        help="Problem difficulty")
    parser.add_argument("--categories", nargs="+", 
                        default=["arithmetic", "algebra"],
                        help="Problem categories")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (default: timestamp)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N iterations")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N iterations")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig.from_args(args)
    
    # Create trainer
    trainer = GRPOTrainer(config)
    
    # Train
    final_metrics = trainer.train(
        n_iterations=config.n_iterations,
        eval_every=config.eval_every,
        save_every=config.save_every
    )
    
    print("\nFinal Results:")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
