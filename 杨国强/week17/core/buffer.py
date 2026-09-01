"""Experience replay buffer for GRPO training."""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class Experience:
    """Single experience tuple."""
    prompt: str
    response: str
    reward: float
    log_prob: float
    group_id: int = 0


class ExperienceBuffer:
    """Buffer for storing experiences during GRPO training.
    
    The buffer organizes experiences by prompt groups to enable
    group-relative advantage computation.
    """
    
    def __init__(self, capacity: int = 10000):
        """Initialize the buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.experiences: List[Experience] = []
        self.group_counter = 0
        
    def add(self, prompt: str, response: str, reward: float, log_prob: float, group_id: Optional[int] = None):
        """Add a single experience to the buffer.
        
        Args:
            prompt: Input prompt
            response: Generated response
            reward: Reward for this response
            log_prob: Log probability of the response
            group_id: Optional group ID (auto-assigned if not provided)
        """
        if group_id is None:
            group_id = self.group_counter
            
        self.experiences.append(Experience(
            prompt=prompt,
            response=response,
            reward=reward,
            log_prob=log_prob,
            group_id=group_id
        ))
        
        # Remove oldest if over capacity
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]
            
    def add_group(self, 
                  prompt: str, 
                  responses: List[str], 
                  rewards: List[float], 
                  log_probs: List[float]):
        """Add a group of responses for the same prompt.
        
        Args:
            prompt: Input prompt
            responses: List of generated responses
            rewards: List of rewards for each response
            log_probs: List of log probabilities for each response
        """
        group_id = self.group_counter
        self.group_counter += 1
        
        for resp, rew, lp in zip(responses, rewards, log_probs):
            self.add(prompt, resp, rew, lp, group_id)
            
    def get_groups(self) -> Dict[int, List[Experience]]:
        """Get experiences organized by group.
        
        Returns:
            Dictionary mapping group_id to list of experiences
        """
        groups = {}
        for exp in self.experiences:
            if exp.group_id not in groups:
                groups[exp.group_id] = []
            groups[exp.group_id].append(exp)
        return groups
    
    def get_group_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors organized by group.
        
        Returns:
            Dictionary with 'rewards', 'log_probs_old', 'advantages', 'masks'
        """
        groups = self.get_groups()
        
        if not groups:
            return {
                'rewards': torch.tensor([], device=device),
                'log_probs_old': torch.tensor([], device=device),
                'advantages': torch.tensor([], device=device),
                'group_sizes': torch.tensor([], device=device)
            }
        
        rewards_list = []
        log_probs_list = []
        
        for gid in sorted(groups.keys()):
            group_exps = groups[gid]
            rewards_list.append([exp.reward for exp in group_exps])
            log_probs_list.append([exp.log_prob for exp in group_exps])
        
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_list, device=device, dtype=torch.float32)
        
        # Compute advantages within each group
        advantages = self._compute_group_advantages(rewards)
        
        return {
            'rewards': rewards,
            'log_probs_old': log_probs_old,
            'advantages': advantages,
            'group_sizes': torch.tensor([len(g) for g in groups.values()], device=device)
        }
    
    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute relative advantages within groups.
        
        Args:
            rewards: Tensor of shape (n_groups, group_size)
            
        Returns:
            advantages: Flat tensor of advantages
        """
        n_groups, group_size = rewards.shape
        
        # Rank responses within each group
        ranks = rewards.argsort(dim=1).argsort(dim=1).float()
        ranks = group_size - ranks  # Higher reward = higher rank
        
        # Convert ranks to relative advantages
        # Normalize to [-0.5, 0.5] then scale
        advantages = (ranks / (group_size - 1) - 0.5) * 2 if group_size > 1 else torch.zeros_like(ranks)
        
        return advantages.view(-1)
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.experiences = []
        self.group_counter = 0
        
    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.experiences)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about the buffer contents.
        
        Returns:
            Dictionary of statistics
        """
        if not self.experiences:
            return {
                'n_experiences': 0,
                'n_groups': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_log_prob': 0.0
            }
        
        rewards = [exp.reward for exp in self.experiences]
        log_probs = [exp.log_prob for exp in self.experiences]
        
        return {
            'n_experiences': len(self.experiences),
            'n_groups': self.group_counter,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_log_prob': np.mean(log_probs)
        }
    
    def sample_recent(self, n: int) -> List[Experience]:
        """Sample n most recent experiences.
        
        Args:
            n: Number of experiences to sample
            
        Returns:
            List of experiences
        """
        return self.experiences[-n:]
