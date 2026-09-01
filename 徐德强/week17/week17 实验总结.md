# Week17 强化学习实验总结 —— GRPO 全量 vs LoRA 训练

> 完成日期：2026-08-23
> 项目目录：`D:\aipy\AI大模型培训部分\week17强化学习和分布式训练\week17_work\grpo_arithmetic`
> 基座模型：`D:\aipy\models\Qwen2.5-0.5B-Instruct`（494M 参数，bf16）
> 运行硬件：NVIDIA GTX 1660 Ti，6GB 显存

---

## 一、实验目标

在消费级显卡上完整复现 **GRPO（Group Relative Policy Optimization）** 强化学习闭环：
用可验证的规则奖励（无需奖励模型）让 Qwen2.5-0.5B 学会"算术正确 + 输出格式"，并对比全量微调与 LoRA 两种方案的效果、资源与时间。

## 二、实验环境

| 项目 | 值 |
|---|---|
| 解释器 | Python 3.12.13（`D:\conda_envs\py312_cuda`） |
| torch | 2.7.1+cu118 |
| transformers | 4.57.6 |
| trl | 0.21.0（项目锁定版本，已降级自 1.10.0） |
| peft / accelerate / datasets | 0.19.1 / 1.13.0 / 4.8.5 |
| 显存 | 6GB（全量峰值 6.37GB，LoRA 峰值 3.36GB） |

## 三、训练配置（`src/train_grpo.py`）

- GRPO：K=8 组内采样、beta=0（不加载参考模型）、epsilon=0.2、temperature=1.0
- 复合奖励：正确分 1.0（宽松解析）+ 格式分 0.2（`<answer>` 标签）
- 难度课程：训练集 1000 题 = L3(50%) + L5(25%) + L2(25%)；L1/L4/L6 留作泛化评估
- 200 步，每步 4 prompt × 8 采样
- 全量 lr 2e-6；LoRA（r=16）lr 2e-4
- 关键修复：显式 bf16 加载（避免 fp16 下溢训废）、关闭 gradient checkpointing（损坏 generate）、trl_compat 补丁

## 四、运行流程

```
1. 基线摸底：probe_baseline.py（6 难度 × 50 题，greedy + pass@8 + informative rate）
2. 全量训练：train_grpo.py（200 步，37 分钟，6.37GB）
3. LoRA 训练：train_grpo.py --lora（200 步，12.7 分钟，3.36GB）
4. 训练后评估：probe_baseline.py --model 各自 ckpt（同 seed 42 配对）
5. 对比分析：compare_results.py（三方对比表 + 训练曲线）
```

## 五、结果对比（同一评估集，seed=42，50 题/难度）

| 难度 | 训练集 | 基线 格式/正确/pass@8 | 全量 格式/正确/pass@8 | LoRA 格式/正确/pass@8 |
|---|---|---|---|---|
| L1 个位加法 | — | 0.38/0.90/1.00 | 1.00/**1.00**/1.00 | 1.00/**1.00**/1.00 |
| L2 两位加减 | √ | 0.44/0.84/1.00 | 1.00/**1.00**/1.00 | 1.00/**0.98**/1.00 |
| L3 三位加减 | √ | 0.68/0.78/0.96 | 1.00/**0.94**/0.96 | 1.00/**0.94**/0.94 |
| L4 表内乘法 | — | 0.50/0.98/1.00 | 1.00/**1.00**/1.00 | 1.00/**1.00**/1.00 |
| L5 两位×一位 | √ | 0.60/0.92/0.96 | 1.00/**0.96**/1.00 | 1.00/**0.92**/0.96 |
| L6 两位×两位 | — | 0.52/0.50/0.70 | 1.00/**0.54**/0.80 | 1.00/**0.48**/0.58 |

**资源对比：**

| 方案 | 训练时间(200步) | 峰值显存 | 可训练参数 | 学习率 |
|---|---|---|---|---|
| 全量 | 2222 秒（37 分钟） | 6.37 GB | 494M（100%） | 2e-6 |
| LoRA(r=16) | 765 秒（12.7 分钟） | **3.36 GB** | ~1.4M（0.28%） | 2e-4 |

## 六、关键结论

1. **格式学习满分且泛化**：全量/LoRA 格式率均 1.00，包括未训练难度——格式是表层行为，RL 极易学会。
2. **训练集内难度提升明显**：L3 0.78→0.94、L2 0.84→1.00；未训难度 L1/L4 到 1.00（泛化，非背题）。
3. **超边界 L6 受限**：0.50→0.54（全量）/0.48（LoRA），pass@8 0.70→0.80/0.58——RL 不能突破 0.5B 能力边界。
4. **LoRA 资源性价比高**：显存减半、时间快约 3 倍，效果与全量接近（本次 L2/L5/L6 全量略胜、L3 持平）。
5. **基线偏高说明**：因本机用 Qwen2.5（明显强于原项目 Qwen2），基线指标整体偏高，压缩了提升空间。

## 七、产物清单（`outputs\`）

- `grpo_ckpt\`：全量训练 checkpoint（含 checkpoint-50/100/150/200 保存点 + 最终模型）
- `grpo_lora_ckpt\`：LoRA adapter checkpoint（含保存点 + 最终 adapter）
- `baseline_probe.json` / `post_train_probe.json` / `post_train_probe_lora.json`：基线+两次评估结果
- `train_log.json` / `train_log_lora.json`：两次训练日志
- `figures\train_curves.png`：全量+LoRA 叠加训练曲线

## 八、本次代码变更

- `src\train_grpo.py`：MODEL_PATH 指向本机；新增保存点（save_strategy=steps/save_steps=50/save_total_limit=3）+ `--resume` 断点续跑；注释更新为 Qwen2.5/6GB
- `src\probe_baseline.py`：MODEL_PATH 指向本机；注释更新
- `src\test_general_dialog.py`：MODEL_PATH 指向本机
- `ARCHITECTURE.md` / `USAGE_GUIDE.md`：按本机实测更新硬件/模型/数据

## 九、遗留问题与建议

- 结果存在训练随机性（temperature=1.0 + seed），多次运行指标有波动，对比以单次实测为准
- 如需更精确结论：可跑多次取均值；长任务可调高温度/加熵正则防探索枯竭
- 完整流程可复用：换任务只需改 make_problem/LEVEL_MIX/reward 三处