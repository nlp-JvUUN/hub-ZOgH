# GRPO系统架构详解

## 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     GRPO训练系统                             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │            1. 数据层 (Data Layer)                   │    │
│  │                                                      │    │
│  │  ┌──────────────┐         ┌──────────────┐        │    │
│  │  │   GSM8K      │   ───>  │  MathDataset │        │    │
│  │  │  (HuggingFace)│         │   (Loader)   │        │    │
│  │  └──────────────┘         └──────────────┘        │    │
│  │         │                         │                 │    │
│  │         │ 问题 + 答案              │ 批处理          │    │
│  │         v                         v                 │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │  format_prompt + collate_fn        │           │    │
│  │  └────────────────────────────────────┘           │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           │ 批次数据                         │
│                           v                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │            2. 模型层 (Model Layer)                  │    │
│  │                                                      │    │
│  │  ┌──────────────┐         ┌──────────────┐        │    │
│  │  │ Policy Model │         │  Ref Model   │        │    │
│  │  │   (训练中)    │         │   (冻结)     │        │    │
│  │  └──────────────┘         └──────────────┘        │    │
│  │         │                         │                 │    │
│  │         │ 生成回答                 │ 参考概率        │    │
│  │         v                         v                 │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │     generate_responses()           │           │    │
│  │  │  (每个问题生成group_size个回答)     │           │    │
│  │  └────────────────────────────────────┘           │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           │ 生成的回答                       │
│                           v                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │          3. 评估层 (Evaluation Layer)               │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │      parse_model_output()          │           │    │
│  │  │    (解析生成的数学答案)             │           │    │
│  │  └────────────────────────────────────┘           │    │
│  │                    │                                │    │
│  │                    v                                │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │       check_answer()                │           │    │
│  │  │    (与真实答案比较)                 │           │    │
│  │  └────────────────────────────────────┘           │    │
│  │                    │                                │    │
│  │                    v                                │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │      compute_rewards()              │           │    │
│  │  │   正确: +1.0  错误: -0.5            │           │    │
│  │  └────────────────────────────────────┘           │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           │ 奖励值                          │
│                           v                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │          4. 优化层 (Optimization Layer)             │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │   compute_grpo_loss()              │           │    │
│  │  │                                     │           │    │
│  │  │   组内标准化:                       │           │    │
│  │  │   A = (R - μ) / σ                  │           │    │
│  │  │                                     │           │    │
│  │  │   策略损失:                         │           │    │
│  │  │   L_policy = -A * log π(r|q)      │           │    │
│  │  │                                     │           │    │
│  │  │   KL惩罚:                           │           │    │
│  │  │   L_kl = β|log π - log πref|      │           │    │
│  │  │                                     │           │    │
│  │  │   总损失:                           │           │    │
│  │  │   L = L_policy + L_kl             │           │    │
│  │  └────────────────────────────────────┘           │    │
│  │                    │                                │    │
│  │                    v                                │    │
│  │  ┌────────────────────────────────────┐           │    │
│  │  │    反向传播 + 梯度更新              │           │    │
│  │  │    optimizer.step()                 │           │    │
│  │  └────────────────────────────────────┘           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 数据流详解

### 单个训练步骤的数据流

```
输入: batch = {
  "questions": ["Q1", "Q2", ...],
  "final_answers": ["A1", "A2", ...]
}

Step 1: 生成阶段
  For each question qi in batch:
    生成 group_size 个回答: [r1, r2, r3, r4]
    记录对数概率:
      - policy_log_probs[i] = [log π(r1|qi), log π(r2|qi), ...]
      - ref_log_probs[i] = [log πref(r1|qi), log πref(r2|qi), ...]

Step 2: 评估阶段
  For each response group [r1, r2, r3, r4]:
    解析答案: [pred1, pred2, pred3, pred4]
    比较真实答案: [is_correct1, is_correct2, is_correct3, is_correct4]
    计算奖励: [R1, R2, R3, R4]  # 1.0 or -0.5

Step 3: 优化阶段
  For each group [R1, R2, R3, R4]:
    计算组内统计:
      μ = mean([R1, R2, R3, R4])
      σ = std([R1, R2, R3, R4])
    
    计算相对优势:
      A1 = (R1 - μ) / σ
      A2 = (R2 - μ) / σ
      A3 = (R3 - μ) / σ
      A4 = (R4 - μ) / σ
    
    计算损失:
      For each i in [1,2,3,4]:
        L_policy_i = -Ai * log π(ri|q)
        L_kl_i = β * |log π(ri|q) - log πref(ri|q)|
        L_i = L_policy_i + L_kl_i
    
    总损失: L_total = mean([L1, L2, L3, L4])

Step 4: 更新
  L_total.backward()
  optimizer.step()
```

## 类结构图

```
┌─────────────────────────────────────────┐
│           GRPOConfig                     │
├─────────────────────────────────────────┤
│ + model_name: str                        │
│ + batch_size: int                        │
│ + learning_rate: float                   │
│ + group_size: int                        │
│ + kl_coef: float                         │
│ + correct_reward: float                  │
│ + incorrect_reward: float                │
│ + ...                                    │
└─────────────────────────────────────────┘
                    │
                    │ 配置
                    v
┌─────────────────────────────────────────┐
│          MathDataset                     │
├─────────────────────────────────────────┤
│ + __init__(split, dataset_name)         │
│ + __len__() -> int                       │
│ + __getitem__(idx) -> Dict               │
│ + extract_answer(text) -> str            │
│ + check_answer(pred, gt) -> bool         │
└─────────────────────────────────────────┘
                    │
                    │ 数据
                    v
┌─────────────────────────────────────────┐
│          GRPOTrainer                     │
├─────────────────────────────────────────┤
│ - config: GRPOConfig                     │
│ - policy_model: AutoModelForCausalLM     │
│ - ref_model: AutoModelForCausalLM        │
│ - optimizer: AdamW                       │
│ - train_dataset: MathDataset             │
│ - val_dataset: MathDataset               │
├─────────────────────────────────────────┤
│ + generate_responses()                   │
│ + compute_rewards()                      │
│ + compute_grpo_loss()                    │
│ + train_step()                           │
│ + evaluate()                             │
│ + train()                                │
│ + save_checkpoint()                      │
└─────────────────────────────────────────┘
```

## 模块依赖图

```
train.py
    │
    ├─> config.py
    │       └─> GRPOConfig
    │
    └─> grpo_trainer.py
            │
            ├─> config.py (GRPOConfig)
            │
            ├─> math_dataset.py
            │       ├─> MathDataset
            │       ├─> format_prompt()
            │       ├─> parse_model_output()
            │       └─> collate_fn()
            │
            └─> transformers
                    ├─> AutoModelForCausalLM
                    └─> AutoTokenizer


inference.py
    │
    └─> math_dataset.py
            ├─> format_prompt()
            └─> parse_model_output()
```

## 核心算法伪代码

### GRPO训练主循环

```python
def train():
    # 初始化
    policy_model = load_model()
    ref_model = load_model()  # 冻结
    optimizer = AdamW(policy_model.parameters())
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in train_loader:
            # 1. 生成阶段
            responses = []
            log_probs = []
            ref_log_probs = []
            
            for question in batch["questions"]:
                group_responses = []
                group_log_probs = []
                group_ref_log_probs = []
                
                for _ in range(group_size):
                    # 策略模型生成
                    response, log_prob = policy_model.generate(question)
                    group_responses.append(response)
                    group_log_probs.append(log_prob)
                    
                    # 参考模型概率
                    ref_log_prob = ref_model.compute_log_prob(question, response)
                    group_ref_log_probs.append(ref_log_prob)
                
                responses.append(group_responses)
                log_probs.append(group_log_probs)
                ref_log_probs.append(group_ref_log_probs)
            
            # 2. 评估阶段
            rewards = []
            for group_responses, ground_truth in zip(responses, batch["answers"]):
                group_rewards = []
                for response in group_responses:
                    predicted = parse_output(response)
                    is_correct = check_answer(predicted, ground_truth)
                    reward = 1.0 if is_correct else -0.5
                    group_rewards.append(reward)
                rewards.append(group_rewards)
            
            # 3. 优化阶段
            total_loss = 0
            
            for group_log_probs, group_ref_log_probs, group_rewards in zip(
                log_probs, ref_log_probs, rewards
            ):
                # 组内标准化
                mean_reward = mean(group_rewards)
                std_reward = std(group_rewards)
                advantages = [(r - mean_reward) / std_reward 
                             for r in group_rewards]
                
                # 计算损失
                for log_prob, ref_log_prob, advantage in zip(
                    group_log_probs, group_ref_log_probs, advantages
                ):
                    # 策略损失
                    policy_loss = -advantage * log_prob.mean()
                    
                    # KL散度
                    kl_div = abs(log_prob.mean() - ref_log_prob.mean())
                    
                    # 总损失
                    loss = policy_loss + kl_coef * kl_div
                    total_loss += loss
            
            # 平均并更新
            avg_loss = total_loss / (batch_size * group_size)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
        
        # 评估
        accuracy = evaluate(policy_model, val_loader)
        print(f"Epoch {epoch}: accuracy = {accuracy}")
```

### GRPO损失计算详解

```python
def compute_grpo_loss(log_probs, ref_log_probs, rewards):
    """
    参数:
        log_probs: List[List[Tensor]]  
            # shape: [batch_size, group_size, seq_len]
        ref_log_probs: List[List[Tensor]]
            # shape: [batch_size, group_size, seq_len]
        rewards: List[List[float]]
            # shape: [batch_size, group_size]
    
    返回:
        loss: Tensor  # scalar
    """
    
    total_loss = 0.0
    num_samples = 0
    
    # 遍历每个问题的组
    for group_lp, group_ref_lp, group_r in zip(
        log_probs, ref_log_probs, rewards
    ):
        # group_lp: [group_size, seq_len]
        # group_r: [group_size]
        
        # 1. 组内标准化奖励
        r_tensor = torch.tensor(group_r)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std() + 1e-8  # 避免除零
        advantages = (r_tensor - r_mean) / r_std  # [group_size]
        
        # 2. 计算每个样本的损失
        for lp, ref_lp, adv in zip(group_lp, group_ref_lp, advantages):
            # lp: [seq_len]
            
            # 平均对数概率
            avg_lp = lp.mean()        # scalar
            avg_ref_lp = ref_lp.mean()  # scalar
            
            # 策略梯度损失
            policy_loss = -adv * avg_lp
            
            # KL散度（简化版）
            kl_div = (avg_lp - avg_ref_lp).abs()
            
            # 组合损失
            sample_loss = policy_loss + kl_coef * kl_div
            
            total_loss += sample_loss
            num_samples += 1
    
    # 平均损失
    return total_loss / num_samples
```

## 关键设计决策

### 1. 为什么使用组内标准化？

**问题**：不同问题的难度不同，绝对奖励值可能差异很大。

**解决**：
```
传统方法: A = R - baseline
GRPO方法: A = (R - mean_group(R)) / std_group(R)

优势:
- 消除不同问题的难度差异
- 减少奖励方差，提高训练稳定性
- 使模型更关注相对质量而非绝对值
```

### 2. 为什么需要参考模型？

**问题**：策略模型可能过度优化奖励，失去原有能力。

**解决**：
```
引入KL散度约束: L_kl = β * KL(π || πref)

作用:
- 防止策略偏离原始模型太远
- 保持模型的通用语言能力
- 避免过拟合到奖励函数
```

### 3. 为什么每个问题生成多个回答？

**问题**：单个样本的梯度估计方差大。

**解决**：
```
组内采样 (group_size=4):
- 同一问题生成4个回答
- 基于组内比较计算优势
- 降低梯度估计方差
- 提供更稳定的训练信号
```

## 性能优化策略

### 内存优化
```python
# 1. 梯度累积
if step % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# 2. 混合精度
with torch.cuda.amp.autocast():
    loss = compute_loss(...)

# 3. 梯度检查点
model.gradient_checkpointing_enable()
```

### 速度优化
```python
# 1. 批量生成
responses = model.generate(
    batch_inputs,
    batch_size=batch_size * group_size
)

# 2. 并行评估
from multiprocessing import Pool
with Pool(num_workers) as pool:
    rewards = pool.map(evaluate_response, responses)

# 3. 缓存参考模型输出
if use_cache:
    ref_log_probs = cache.get(question, response)
```

## 总结

GRPO系统通过精心设计的架构，实现了稳定且有效的强化学习训练：

1. **模块化设计**：数据、模型、评估、优化层次清晰
2. **鲁棒性**：组内标准化、KL约束、梯度裁剪
3. **可扩展性**：支持自定义奖励、多任务、分布式
4. **易用性**：完整的配置、日志、检查点系统

---

**文档版本**: 1.0  
**最后更新**: 2024
