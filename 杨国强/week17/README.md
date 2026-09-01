# GRPO Math Trainer

A reinforcement learning system based on **Group Relative Policy Optimization (GRPO)** to improve language model performance on mathematical reasoning tasks.

## Overview

This project implements GRPO, a policy optimization algorithm that uses group-relative ranking to compute advantages. Unlike standard policy gradient methods that compare to a baseline, GRPO compares responses within groups, enabling more efficient learning for language model alignment.

## Features

- **GRPO Algorithm**: Group-relative policy optimization with PPO-style clipping
- **Math Problem Environment**: Diverse set of problems across 8 categories and 4 difficulty levels
- **Multi-component Rewards**: Accuracy, format, step-by-step, and partial credit rewards
- **Multiple Backend Support**: Mock, OpenAI, and Anthropic API policies
- **Comprehensive Logging**: Metrics tracking, checkpointing, and evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Run with mock policy (no API needed)
python train.py --n_iterations 50 --batch_size 8 --group_size 4

# Run with OpenAI API
python train.py --policy_type openai --model_name gpt-4 --n_iterations 100

# Run with Anthropic API
python train.py --policy_type anthropic --model_name claude-3-opus-20240229
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --n_problems 100 --verbose

# Show example problems and responses
python evaluate.py --show_examples 5
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_iterations` | 100 | Number of training iterations |
| `--batch_size` | 16 | Problems per iteration |
| `--group_size` | 8 | Responses per prompt in GRPO |
| `--learning_rate` | 1e-5 | Learning rate |
| `--kl_coef` | 0.1 | KL divergence penalty |
| `--clip_ratio` | 0.2 | PPO clipping ratio |

### Problem Categories

Available categories: `arithmetic`, `algebra`, `geometry`, `calculus`, `word_problems`, `number_theory`, `combinatorics`, `probability`

Example:
```bash
python train.py --categories arithmetic algebra geometry --difficulty medium
```

## Project Structure

```
grpo_math_trainer/
├── core/
│   ├── grpo.py           # GRPO algorithm implementation
│   ├── policy.py         # Policy model interface
│   └── buffer.py         # Experience replay buffer
├── envs/
│   └── math_env.py       # Math problem environment
├── rewards/
│   └── reward_model.py   # Multi-component reward system
├── utils/
│   ├── config.py         # Configuration management
│   └── logger.py         # Logging utilities
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
└── requirements.txt
```

## How GRPO Works

1. **Group Sampling**: For each prompt, sample G responses from the current policy
2. **Reward Computation**: Calculate rewards for all responses using the reward model
3. **Advantage Estimation**: Compute advantages based on relative ranking within each group
4. **Policy Update**: Update policy using clipped policy gradient, maximizing relative advantage

### Key Algorithm Details

```
For each group of G responses:
  - Rank responses by reward
  - Higher-ranked responses get positive advantages
  - Lower-ranked responses get negative advantages

Policy Update:
  loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

## Reward Components

1. **Accuracy (1.0 weight)**: Correctness of the final answer
2. **Format (0.1 weight)**: Proper solution structure and notation
3. **Step Penalty**: Penalty for missing intermediate steps
4. **Length Penalty**: Slight penalty for inappropriate response length
5. **Partial Credit (0.5 weight)**: Credit for partially correct solutions

## Examples

### Basic Training

```python
from train import GRPOTrainer
from utils.config import TrainingConfig

config = TrainingConfig(
    n_iterations=100,
    batch_size=16,
    group_size=8,
    policy_type="mock"
)

trainer = GRPOTrainer(config)
trainer.train(n_iterations=100)
```

### Custom Evaluation

```python
from evaluate import MathEvaluator, create_evaluator

evaluator = create_evaluator(args)
results = evaluator.evaluate(n_problems=200, verbose=True)
evaluator.print_results(results)
```

## License

MIT License

## References

- Schulman, J. et al. "Proximal Policy Optimization Algorithms" (2017)
- DeepSeek-AI "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs" (2025)
