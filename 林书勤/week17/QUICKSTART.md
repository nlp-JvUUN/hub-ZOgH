# 快速开始指南

## 5分钟上手GRPO训练

### 步骤1：安装依赖

```bash
pip install torch transformers datasets numpy tqdm
```

或使用requirements文件：

```bash
pip install -r requirements.txt
```

### 步骤2：测试环境

```bash
python test_components.py
```

如果看到 "All tests passed!"，说明环境配置正确。

### 步骤3：开始训练

#### 快速测试（CPU，小数据集）
```bash
python train.py --device cpu --batch_size 1 --dataset_split "train[:10]" --val_split "test[:5]" --num_epochs 1
```

这将在几分钟内完成，帮你验证代码是否工作正常。

#### GPU训练（推荐）
```bash
python train.py
```

默认配置会使用：
- 1000个训练样本
- 100个验证样本
- 训练3轮
- 大约需要30-60分钟（取决于GPU）

### 步骤4：使用训练好的模型

```bash
python inference.py --model_path ./grpo_math_model/final --question "If John has 5 apples and buys 3 more, how many apples does he have in total?"
```

## 命令速查表

### 训练相关

```bash
# 默认训练
python train.py

# 自定义训练
python train.py --batch_size 2 --num_epochs 5 --learning_rate 5e-6

# CPU训练（无GPU时）
python train.py --device cpu --batch_size 1

# 快速测试
python train.py --dataset_split "train[:100]" --num_epochs 1
```

### 推理相关

```bash
# 单问题求解
python inference.py --model_path ./grpo_math_model/final --question "YOUR_QUESTION"

# 交互模式
python inference.py --model_path ./grpo_math_model/final

# 调整温度（更保守/更有创意）
python inference.py --temperature 0.3  # 更保守
python inference.py --temperature 1.0  # 更有创意
```

## 常用配置组合

### 内存受限（4GB VRAM）
```bash
python train.py \
  --batch_size 1 \
  --group_size 2 \
  --dataset_split "train[:500]"
```

### 标准配置（8GB VRAM）
```bash
python train.py \
  --batch_size 4 \
  --group_size 4 \
  --dataset_split "train[:2000]"
```

### 完整训练（16GB+ VRAM）
```bash
python train.py \
  --batch_size 8 \
  --group_size 8 \
  --dataset_split "train" \
  --val_split "test" \
  --num_epochs 5
```

## 训练监控

训练过程中你会看到：

```
Training Epoch 1: 100%|██████████| 250/250 [10:23<00:00, loss=0.7234, reward=0.15]
```

关键指标：
- **loss**: 应该逐渐下降（0.8 → 0.3）
- **reward**: 应该逐渐上升（0.0 → 0.5+）

## 预期结果

- **初始准确率**: 30-40%
- **训练后（1000样本）**: 45-55%
- **完整训练**: 60-70%

## 下一步

详细文档请查看 `readme` 文件，包含：
- 完整的算法原理
- 详细的参数说明
- 性能优化建议
- 常见问题解答

## 问题排查

### 问题：CUDA out of memory

**解决**：
```bash
python train.py --batch_size 1 --group_size 2
```

### 问题：训练太慢

**解决**：
```bash
python train.py --dataset_split "train[:200]"
```

### 问题：准确率不提升

**解决**：
1. 增加训练数据：`--dataset_split "train[:5000]"`
2. 增加训练轮数：`--num_epochs 5`
3. 调整学习率：`--learning_rate 5e-6`

---

需要帮助？请查看 `readme` 文件获取详细信息！
