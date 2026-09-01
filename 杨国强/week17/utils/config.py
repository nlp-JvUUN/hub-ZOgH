"""Configuration management for GRPO training."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class TrainingConfig:
    """Main configuration for GRPO training."""
    
    # Training parameters
    n_iterations: int = 100
    batch_size: int = 16
    group_size: int = 8
    n_epochs: int = 4
    learning_rate: float = 1e-5
    
    # GRPO parameters
    kl_coef: float = 0.1
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Model parameters
    policy_type: str = "mock"  # mock, openai, anthropic
    model_name: str = "gpt-4"
    temperature: float = 0.8
    max_tokens: int = 512
    
    # Environment parameters
    difficulty: str = "medium"  # easy, medium, hard, expert
    problem_categories: List[str] = field(default_factory=lambda: ["arithmetic", "algebra"])
    seed: int = 42
    buffer_capacity: int = 10000
    
    # Reward parameters
    require_step_by_step: bool = True
    
    # Output parameters
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    experiment_name: Optional[str] = None
    eval_every: int = 5
    save_every: int = 10
    log_level: str = "INFO"
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.experiment_name is None:
            self.experiment_name = f"grpo_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Make output directories
        self.log_dir = os.path.join(self.log_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
        
    @classmethod
    def from_args(cls, args) -> 'TrainingConfig':
        """Create config from parsed command line arguments."""
        config_dict = {}
        
        # Map of arg names to config names
        arg_to_config = {
            'n_iterations': 'n_iterations',
            'batch_size': 'batch_size',
            'group_size': 'group_size',
            'n_epochs': 'n_epochs',
            'learning_rate': 'learning_rate',
            'kl_coef': 'kl_coef',
            'clip_ratio': 'clip_ratio',
            'entropy_coef': 'entropy_coef',
            'policy_type': 'policy_type',
            'model_name': 'model_name',
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'difficulty': 'difficulty',
            'categories': 'problem_categories',
            'seed': 'seed',
            'log_dir': 'log_dir',
            'checkpoint_dir': 'checkpoint_dir',
            'experiment_name': 'experiment_name',
            'eval_every': 'eval_every',
            'save_every': 'save_every',
        }
        
        for arg_name, config_name in arg_to_config.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_name] = getattr(args, arg_name)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from a JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save(self, path: str):
        """Save config to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def update(self, **kwargs):
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config(path: str) -> TrainingConfig:
    """Load configuration from file.
    
    Args:
        path: Path to config file
        
    Returns:
        TrainingConfig instance
    """
    return TrainingConfig.load(path)


def save_config(config: TrainingConfig, path: str):
    """Save configuration to file.
    
    Args:
        config: TrainingConfig instance
        path: Path to save config
    """
    config.save(path)


def merge_configs(base: TrainingConfig, override: Dict[str, Any]) -> TrainingConfig:
    """Merge override dictionary into base config.
    
    Args:
        base: Base configuration
        override: Dictionary of values to override
        
    Returns:
        Merged configuration
    """
    base_dict = base.to_dict()
    base_dict.update(override)
    return TrainingConfig.from_dict(base_dict)
