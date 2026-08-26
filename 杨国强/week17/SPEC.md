# GRPO Math Training System

## Project Overview

- **Project Name**: GRPO Math Trainer
- **Type**: Reinforcement Learning Training System
- **Core Functionality**: Implement Group Relative Policy Optimization (GRPO) to improve model performance on mathematical reasoning tasks
- **Target Users**: ML researchers and engineers working on LLM mathematical capabilities

## Architecture

```
grpo_math_trainer/
├── core/
│   ├── __init__.py
│   ├── grpo.py              # GRPO algorithm implementation
│   ├── policy.py            # Policy network management
│   └── buffer.py            # Experience replay buffer
├── envs/
│   ├── __init__.py
│   └── math_env.py          # Math problem environment
├── rewards/
│   ├── __init__.py
│   └── reward_model.py      # Reward calculation
├── utils/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   └── logger.py            # Logging utilities
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── requirements.txt
└── README.md
```

## Functionality Specification

### 1. GRPO Algorithm Core (`core/grpo.py`)

**Group Relative Policy Optimization** is inspired by PPO but uses relative ranking within a group:

- **Group Sampling**: For each prompt, sample G responses from current policy
- **Advantage Estimation**: Calculate advantages using relative ranking within group
- **Policy Update**: Update policy to maximize relative advantage
- **KL Divergence Constraint**: Penalize large policy changes

**Key Components**:
- `GRPOConfig`: Hyperparameters (learning rate, group size, KL penalty, etc.)
- `GRPOAgent`: Main algorithm class with update step
- `compute_advantages()`: Calculate group-relative advantages
- `update_policy()`: Perform policy gradient update

### 2. Policy Management (`core/policy.py`)

- Interface for language model (supports OpenAI, Anthropic, local models)
- Response generation with temperature/top-k sampling
- Log probability tracking for policy gradient calculations

### 3. Experience Buffer (`core/buffer.py`)

- Store (prompt, response, reward, log_prob) tuples
- Group-based organization for GRPO
- Efficient batch retrieval for training

### 4. Math Environment (`envs/math_env.py`)

- Problem bank with various difficulty levels
- Categories: arithmetic, algebra, calculus, word problems
- `generate_prompt()`: Create formatted prompts
- `check_answer()`: Validate responses

### 5. Reward Model (`rewards/reward_model.py`)

**Multi-component reward system**:
- **Accuracy Reward**: Binary (correct/incorrect) or partial credit
- **Format Reward**: Reward for proper solution format
- **Step-by-step Reward**: Reward intermediate correct steps
- **Length Penalty**: Slight penalty for overly verbose solutions

### 6. Training Loop (`train.py`)

**Training Process**:
1. Sample batch of math problems
2. For each problem, generate G responses (group size)
3. Calculate rewards for all responses
4. Compute GRPO advantages
5. Update policy
6. Log metrics and save checkpoints

**Metrics to Track**:
- Mean reward per iteration
- Accuracy rate
- KL divergence
- Response length distribution
- Per-category performance

### 7. Evaluation (`evaluate.py`)

- Evaluate on held-out test set
- Generate detailed performance reports
- Compare against baseline

## Configuration Options

```python
# Training hyperparameters
learning_rate: float = 1e-5
group_size: int = 8          # G in GRPO
kl_coef: float = 0.1         # KL divergence penalty
clip_ratio: float = 0.2      # PPO-style clipping
n_epochs: int = 4            # Update epochs per batch
batch_size: int = 16         # Problems per batch

# Model settings
model_name: str = "gpt-4"
temperature: float = 0.8
max_tokens: int = 512

# Environment
problem_difficulty: str = "medium"
problem_categories: List[str] = ["arithmetic", "algebra"]
n_train_problems: int = 1000
n_eval_problems: int = 200
```

## Acceptance Criteria

1. **GRPO Implementation**
   - [x] Group sampling generates G diverse responses per prompt
   - [x] Advantages correctly computed using relative ranking
   - [x] Policy updates respect KL constraint
   - [x] Training converges over iterations

2. **Math Environment**
   - [x] Problems cover multiple categories and difficulty levels
   - [x] Answer checking is accurate
   - [x] Prompts are properly formatted

3. **Reward System**
   - [x] Rewards reflect solution quality
   - [x] Multi-component rewards combine appropriately
   - [x] Reward signal is informative for learning

4. **Training**
   - [x] Training loop runs without errors
   - [x] Metrics are logged correctly
   - [x] Model checkpointing works
   - [x] Can resume training from checkpoint

5. **Evaluation**
   - [x] Evaluation produces accuracy metrics
   - [x] Results are logged and can be visualized

## Dependencies

- torch >= 2.0
- numpy >= 1.24
- openai >= 1.0 (or anthropic)
- tiktoken (tokenization)
- tensorboard (logging)
- tqdm (progress bars)
