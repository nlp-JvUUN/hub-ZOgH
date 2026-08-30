# Week 17 作业文件索引

## 🚀 快速开始

**第一次使用？从这里开始：**
1. 阅读 [`QUICKSTART.md`](QUICKSTART.md) - 5分钟快速上手
2. 运行 `python test_components.py` - 测试环境
3. 运行 `python train.py --dataset_split "train[:10]" --num_epochs 1` - 快速测试训练

## 📁 文件导航

### 核心代码（必读）

| 文件 | 描述 | 优先级 |
|------|------|--------|
| [`train.py`](train.py) | **训练主程序** - 开始训练从这里 | ⭐⭐⭐ |
| [`inference.py`](inference.py) | **推理脚本** - 使用训练好的模型 | ⭐⭐⭐ |
| [`grpo_trainer.py`](grpo_trainer.py) | **GRPO训练器** - 核心算法实现 | ⭐⭐⭐ |
| [`math_dataset.py`](math_dataset.py) | 数据集处理 | ⭐⭐ |
| [`config.py`](config.py) | 配置文件 | ⭐⭐ |

### 文档（推荐阅读）

| 文件 | 描述 | 适合人群 |
|------|------|---------|
| [`QUICKSTART.md`](QUICKSTART.md) | **5分钟快速开始** | 所有人 ⭐⭐⭐ |
| [`readme`](readme) | **完整文档和教程** | 详细学习 ⭐⭐⭐ |
| [`作业完成报告.md`](作业完成报告.md) | 作业总结报告 | 评审/总结 ⭐⭐⭐ |
| [`SUMMARY.md`](SUMMARY.md) | 技术总结 | 快速了解 ⭐⭐ |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 架构详解 | 深入理解 ⭐⭐ |

### 辅助文件

| 文件 | 描述 |
|------|------|
| [`test_components.py`](test_components.py) | 组件测试脚本 |
| [`requirements.txt`](requirements.txt) | Python依赖清单 |
| `分布式训练.pptx` | 学习资料 |
| `强化学习.pptx` | 学习资料 |

## 📖 阅读路径建议

### 路径1：快速使用者（15分钟）

```
1. QUICKSTART.md (5分钟)
   └─ 了解如何快速开始

2. 运行测试 (2分钟)
   └─ python test_components.py

3. 快速训练 (5分钟)
   └─ python train.py --dataset_split "train[:10]" --num_epochs 1

4. 推理测试 (3分钟)
   └─ python inference.py --model_path ./grpo_math_model/final
```

### 路径2：学习者（1小时）

```
1. QUICKSTART.md (5分钟)
   └─ 快速了解项目

2. SUMMARY.md (15分钟)
   └─ 理解GRPO算法和整体架构

3. readme (30分钟)
   └─ 详细了解实现细节和使用方法

4. ARCHITECTURE.md (10分钟)
   └─ 深入理解系统架构
```

### 路径3：开发者（2小时）

```
1. QUICKSTART.md (5分钟)
   └─ 快速上手

2. ARCHITECTURE.md (20分钟)
   └─ 理解系统架构

3. 阅读核心代码 (1小时)
   ├─ config.py (10分钟)
   ├─ math_dataset.py (15分钟)
   ├─ grpo_trainer.py (30分钟) - 核心
   └─ train.py (5分钟)

4. 运行和调试 (35分钟)
   ├─ 测试组件 (5分钟)
   ├─ 训练实验 (20分钟)
   └─ 推理测试 (10分钟)
```

## 🎯 常用命令速查

### 训练相关

```bash
# 快速测试（CPU，小数据）
python train.py --device cpu --batch_size 1 --dataset_split "train[:10]" --num_epochs 1

# 标准训练（GPU）
python train.py

# 完整训练
python train.py --batch_size 8 --num_epochs 5 --dataset_split "train"

# 自定义配置
python train.py --learning_rate 5e-6 --group_size 8 --kl_coef 0.05
```

### 推理相关

```bash
# 单问题推理
python inference.py --model_path ./grpo_math_model/final --question "YOUR_QUESTION"

# 交互模式
python inference.py --model_path ./grpo_math_model/final

# 调整温度
python inference.py --temperature 0.3  # 更保守
python inference.py --temperature 1.0  # 更有创意
```

### 测试相关

```bash
# 测试环境和组件
python test_components.py

# 查看帮助
python train.py --help
python inference.py --help
```

## 📊 项目统计

### 代码统计

```
├── 核心代码:      1037 行
├── 测试代码:       186 行
├── 配置代码:        62 行
└── 总计:         1285 行
```

### 文档统计

```
├── README:       ~5000 字
├── SUMMARY:      ~3000 字
├── ARCHITECTURE: ~3500 字
├── QUICKSTART:   ~1500 字
├── 完成报告:      ~4000 字
└── 总计:        ~17000 字
```

### 文件统计

```
├── Python文件:     6 个
├── 文档文件:       6 个
├── 配置文件:       1 个
├── 学习资料:       2 个
└── 总计:          15 个
```

## 🔍 功能索引

### 数据处理
- 加载GSM8K数据集 → `math_dataset.py` > `MathDataset.__init__()`
- 提取答案 → `math_dataset.py` > `extract_answer()`
- 验证答案 → `math_dataset.py` > `check_answer()`
- 格式化提示 → `math_dataset.py` > `format_prompt()`

### 模型训练
- GRPO训练器 → `grpo_trainer.py` > `GRPOTrainer`
- 生成回答 → `grpo_trainer.py` > `generate_responses()`
- 计算奖励 → `grpo_trainer.py` > `compute_rewards()`
- 计算损失 → `grpo_trainer.py` > `compute_grpo_loss()`
- 训练步骤 → `grpo_trainer.py` > `train_step()`

### 模型推理
- 推理器 → `inference.py` > `MathSolver`
- 单问题求解 → `inference.py` > `solve()`
- 批量求解 → `inference.py` > `batch_solve()`

### 配置管理
- 训练配置 → `config.py` > `GRPOConfig`
- 命令行参数 → `train.py` > `parse_args()`

## 🛠️ 开发指南

### 修改训练参数

编辑 `config.py`:
```python
@dataclass
class GRPOConfig:
    batch_size: int = 4        # 修改批次大小
    learning_rate: float = 1e-5  # 修改学习率
    group_size: int = 4         # 修改组大小
    kl_coef: float = 0.1        # 修改KL系数
```

### 使用自定义数据集

编辑 `math_dataset.py`:
```python
def __init__(self, split: str, dataset_name: str):
    if dataset_name == "my_dataset":
        # 加载你的数据集
        self.dataset = load_dataset("path/to/your/dataset")
```

### 自定义奖励函数

编辑 `grpo_trainer.py`:
```python
def compute_rewards(self, responses, ground_truths):
    # 实现你的奖励逻辑
    # 可以考虑：
    # - 推理过程质量
    # - 部分正确奖励
    # - 格式规范性
    pass
```

## 📚 参考资料

### 算法相关
- GRPO论文：Group Relative Policy Optimization
- PPO论文：Proximal Policy Optimization Algorithms
- 强化学习教材：Reinforcement Learning: An Introduction

### 数据集相关
- GSM8K数据集：https://huggingface.co/datasets/gsm8k
- GSM8K论文：Training Verifiers to Solve Math Word Problems

### 模型相关
- Qwen2.5模型：https://huggingface.co/Qwen
- Transformers库：https://huggingface.co/docs/transformers

## ❓ 常见问题快速链接

| 问题 | 解决方案位置 |
|------|-------------|
| CUDA内存不足 | `readme` > 常见问题 > Q1 |
| 训练太慢 | `readme` > 常见问题 > Q2 |
| 准确率不提升 | `readme` > 常见问题 > Q3 |
| 自定义数据集 | `readme` > 常见问题 > Q4 |
| 环境安装 | `QUICKSTART.md` > 步骤1 |
| 性能优化 | `readme` > 性能优化建议 |

## 🎓 学习检查清单

使用本项目学习，确保你理解了：

### 理论层面
- [ ] GRPO算法原理
- [ ] 组相对优化的优势
- [ ] KL散度在RL中的作用
- [ ] 强化学习基本概念

### 实践层面
- [ ] 如何处理数学问题数据
- [ ] 如何设计奖励函数
- [ ] 如何实现RL训练循环
- [ ] 如何评估模型性能

### 工程层面
- [ ] 模块化代码设计
- [ ] 配置管理最佳实践
- [ ] 错误处理和日志
- [ ] 文档编写规范

## 📞 获取帮助

1. **查看文档**：大多数问题在文档中都有答案
2. **运行测试**：`python test_components.py` 可以诊断常见问题
3. **查看日志**：训练日志包含详细的错误信息
4. **检查配置**：确保所有参数都在合理范围内

## 🎉 下一步

完成基础训练后，可以尝试：

1. **调优参数**：实验不同的超参数组合
2. **扩展数据**：使用更多训练数据
3. **改进奖励**：设计更细粒度的奖励函数
4. **多任务学习**：训练解决多种类型的问题
5. **模型融合**：结合多个模型的预测

---

**版本**: 1.0  
**最后更新**: 2024  
**维护者**: 林书勤

**快速联系**:
- 📖 完整文档: [`readme`](readme)
- 🚀 快速开始: [`QUICKSTART.md`](QUICKSTART.md)
- 📊 技术总结: [`SUMMARY.md`](SUMMARY.md)
- 🏗️ 架构详解: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- 📝 完成报告: [`作业完成报告.md`](作业完成报告.md)
