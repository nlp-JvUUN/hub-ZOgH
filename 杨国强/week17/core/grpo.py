"""GRPO (Group Relative Policy Optimization) Algorithm Implementation.

This module implements the GRPO algorithm which uses group-relative ranking
to compute advantages, inspired by PPO but with a focus on relative comparison
within each prompt group.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm."""
    learning_rate: float = 1e-5
    group_size: int = 8  # Number of responses per prompt
    kl_coef: float = 0.1  # KL divergence penalty coefficient
    clip_ratio: float = 0.2  # Clipping ratio for policy gradient
    n_epochs: int = 4  # Number of epochs per update
    batch_size: int = 16  # Batch size for updates
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 1.0  # Gradient clipping norm
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GRPO:
    """Group Relative Policy Optimization algorithm.
    
    GRPO improves upon standard policy gradient methods by computing advantages
    based on relative ranking within groups of responses to the same prompt.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = config.device
        
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages using ranking.
        
        For each group of G responses, we compute advantages based on
        relative ranking - responses ranked higher get positive advantages,
        responses ranked lower get negative advantages.
        
        Args:
            rewards: Tensor of shape (n_groups, group_size) containing rewards
            
        Returns:
            advantages: Tensor of shape (n_groups * group_size,) with advantages
        """
        n_groups, group_size = rewards.shape
        
        # Rank responses within each group (higher reward = higher rank)
        ranks = rewards.argsort(dim=1).argsort(dim=1).float()  # 0 to G-1
        ranks = group_size - ranks  # Now higher reward = higher rank (G to 1)
        
        # Normalize ranks to [0, 1]
        ranks = (ranks - 1) / (group_size - 1) if group_size > 1 else torch.zeros_like(ranks)
        
        # Convert to advantages: center around 0, scale by rank difference
        advantages = ranks - 0.5  # Centered at 0
        advantages = advantages * 2  # Scale to [-1, 1]
        
        # Flatten for output
        return advantages.view(-1)
    
    def compute_kl_divergence(self, 
                               log_probs_old: torch.Tensor,
                               log_probs_new: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between old and new policies.
        
        Args:
            log_probs_old: Old policy log probabilities
            log_probs_new: New policy log probabilities
            
        Returns:
            kl_div: KL divergence
        """
        return (log_probs_old.exp() * (log_probs_old - log_probs_new)).sum(dim=-1).mean()
    
    def compute_policy_loss(self,
                           log_probs: torch.Tensor,
                           log_probs_old: torch.Tensor,
                           advantages: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute clipped policy gradient loss.
        
        Uses PPO-style clipping to prevent too-large policy updates.
        
        Args:
            log_probs: New policy log probabilities
            log_probs_old: Old policy log probabilities
            advantages: Computed advantages
            mask: Optional mask for valid entries
            
        Returns:
            loss: Policy loss
            metrics: Dictionary of loss components
        """
        # Probability ratio
        ratio = torch.exp(log_probs - log_probs_old)
        
        # Clipped objective
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages.unsqueeze(-1)
        
        # Take minimum (PPO-style)
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply mask if provided
        if mask is not None:
            policy_loss = (policy_loss * mask).sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()
        
        # Compute KL for metrics
        with torch.no_grad():
            kl_div = self.compute_kl_divergence(log_probs_old, log_probs)
        
        return policy_loss, {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item()
        }
    
    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus for exploration.
        
        Args:
            logits: Policy logits
            
        Returns:
            entropy: Negative entropy (to be added to loss)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -self.config.entropy_coef * entropy
    
    def update(self,
               model: nn.Module,
               optimizer: torch.optim.Optimizer,
               data: Dict[str, torch.Tensor],
               epoch: int) -> Dict[str, float]:
        """Perform one update step of GRPO.
        
        Args:
            model: Policy model
            optimizer: Optimizer
            data: Dictionary containing:
                - 'log_probs_old': Old log probabilities (n_samples, seq_len)
                - 'log_probs_new': New log probabilities (n_samples, seq_len)
                - 'rewards': Rewards for each response (n_groups, group_size)
                - 'advantages': Precomputed advantages
                - 'masks': Padding masks (optional)
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary of training metrics
        """
        model.train()
        
        # Unpack data
        log_probs_new = data['log_probs_new']
        log_probs_old = data['log_probs_old']
        advantages = data['advantages']
        masks = data.get('masks', None)
        
        # Reshape for grouped processing
        n_groups = advantages.shape[0]
        advantages = advantages.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        log_probs_new = log_probs_new.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        # Compute losses
        policy_loss, policy_metrics = self.compute_policy_loss(
            log_probs_new, log_probs_old, advantages, masks
        )
        
        # Get logits for entropy (assuming model returns logits)
        with torch.no_grad():
            dummy_logits = log_probs_new  # Use log_probs as proxy
        entropy_loss = self.compute_entropy_loss(dummy_logits)
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (only if model has trainable parameters)
        if self.config.max_grad_norm > 0:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            except (AttributeError, TypeError):
                pass  # Skip if model doesn't support parameters()
        
        optimizer.step()
        
        # Aggregate metrics
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl_div": policy_metrics["kl_div"],
            "ratio_mean": policy_metrics["ratio_mean"],
            "epoch": epoch
        }
        
        return metrics
    
    def group_responses_by_prompt(self, 
                                    prompts: List[str],
                                    responses: List[str],
                                    rewards: List[float]) -> List[List[Dict]]:
        """Group responses by their source prompt.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            rewards: List of rewards
            
        Returns:
            groups: List of groups, each containing dicts with response data
        """
        prompt_to_group = defaultdict(list)
        
        for i, prompt in enumerate(prompts):
            prompt_to_group[prompt].append({
                'response': responses[i],
                'reward': rewards[i],
                'index': i
            })
        
        return list(prompt_to_group.values())
