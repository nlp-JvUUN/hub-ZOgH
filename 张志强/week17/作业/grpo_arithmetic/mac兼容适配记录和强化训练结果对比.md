# Mac 兼容适配：练习 vs 课件代码差异总结

针对 Mac（Apple M1 Pro, macOS 26.2, 无 CUDA）做了以下适配修改。

---

## 1. `trl_compat.py` — 新增 MPS 后端禁用补丁

**问题**：macOS 26.x + torch 2.13.0 的 MPS 后端处于"编译了但运行时不可用"的损坏状态（`_has_mps=True` 但 `_mps_is_available()=False`）。DataLoader 的 `pin_memory` worker 线程仅检查 `_has_mps` 就尝试 `.to('mps')`，触发 `MetalShaderLibrary::getFunctionNames()` 空指针解引用，直接 segfault。

**修改**：新增 `_patch_disable_mps()` 函数，在 import 阶段强制将 `torch._C._has_mps` 设为 `False`，让所有设备检测逻辑回退到 CPU。

```python
def _patch_disable_mps():
    import torch._C as C
    if hasattr(C, "_has_mps"):
        C._has_mps = False
```

并在模块末尾的补丁调用区追加 `_patch_disable_mps()`。

---

## 2. `train_grpo.py` — 精度、设备、路径三处适配


### 2.1 模型加载精度：bfloat16 → float32

Mac CPU 不支持 bfloat16 运算。课件代码在 CUDA 环境用 `bfloat16`，练习改为 `float32` 保证 CPU 上的数值稳定性。

```diff
- model_init_kwargs={"torch_dtype": "bfloat16"},
+ model_init_kwargs={"torch_dtype": "float32"},
```

### 2.2 混合精度：关闭 bf16/fp16

课件代码开启 `bf16=True`（CUDA AMP），Mac 上无 CUDA AMP 支持，两项均关闭：

```diff
- bf16=True,
+ bf16=False,
+ fp16=False,
```

### 2.3 移除 max_prompt_length

```diff
- max_prompt_length=128,
```

删除了该参数（Mac 上跑 LoRA 时 prompt 较短，不需要限制）。

### 2.4 GPU 显存打印：条件守卫

课件代码无条件调用 `torch.cuda.max_memory_allocated()`，Mac 上无 CUDA 会报错。练习版本加了 `torch.cuda.is_available()` 守卫：

```diff
- peak_gb = torch.cuda.max_memory_allocated() / 1024**3
  print(f"\n训练完成。checkpoint: {ckpt_dir}")
  print(f"训练日志: {log_path}")
+ if torch.cuda.is_available():
+     peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
+     print(f"GPU 峰值显存: {peak_gb:.2f} GB")
```

---

## 3. `probe_baseline.py` — 运行时设备自动检测

这是改动最大的文件，核心是引入 `resolve_runtime()` 函数实现跨设备自适应。

### 3.1 模型路径

同 `train_grpo.py`，Windows 路径改为 Mac 路径。

### 3.2 新增 `resolve_runtime()` 函数

按 CUDA → MPS → CPU 优先级选择设备和对应精度：

```python
def resolve_runtime():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32
```

### 3.3 模型加载：从硬编码 CUDA 改为动态设备

课件代码硬编码 `device_map="cuda"` 和 `dtype=torch.bfloat16`，改为根据 `resolve_runtime()` 的返回值动态构建加载参数，非 CUDA 设备在加载后手动 `.to(runtime_device)`：

```diff
  # 基座模型加载（LoRA 分支）
- base = AutoModelForCausalLM.from_pretrained(
-     MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"
- )
+ base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
+ if runtime_device.type != "cuda":
+     base = base.to(runtime_device)
```

非 LoRA 分支同理。

### 3.4 GPU 显存打印：条件守卫

与 `train_grpo.py` 相同的 CUDA 守卫处理。

---

## 4. `compare_results.py` — 输出文件路径适配

Mac 上只跑了 LoRA 训练（全量微调 OOM），因此对比脚本读取的文件从全量训练产物改为 LoRA 训练产物：

```diff
- with open(OUT / "post_train_probe.json", encoding="utf-8") as f:
+ with open(OUT / "post_train_probe_lora.json", encoding="utf-8") as f:
      post_full = json.load(f)
- with open(OUT / "train_log.json", encoding="utf-8") as f:
+ with open(OUT / "train_log_lora.json", encoding="utf-8") as f:
      log_full = json.load(f)
```

---

## 修改分类汇总

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `trl_compat.py` | 新增补丁 | 禁用损坏的 MPS 后端，防止 segfault |
| `train_grpo.py` | 精度适配 | bfloat16 → float32（CPU 不支持 bf16） |
| `train_grpo.py` | 混合精度 | 关闭 bf16/fp16（无 CUDA AMP） |
| `train_grpo.py` | 参数裁剪 | 移除 max_prompt_length |
| `train_grpo.py` | 安全守卫 | GPU 显存打印加 CUDA 可用性判断 |
| `probe_baseline.py` | 路径适配 | Windows 路径 → Mac 路径 |
| `probe_baseline.py` | 设备抽象 | 新增 `resolve_runtime()` 实现跨设备自适应 |
| `probe_baseline.py` | 加载逻辑 | 硬编码 `device_map="cuda"` → 动态设备选择 |
| `probe_baseline.py` | 安全守卫 | GPU 显存打印加 CUDA 可用性判断 |
| `compare_results.py` | 产物路径 | 从全量训练产物改为 LoRA 训练产物 |
| `test_general_dialog.py` | 无修改 | — |

---

# 输出产物总结（`grpo_arithmetic/outputs/`）

## 目录结构

```
outputs/
├── baseline_probe.json          # 训练前基座模型探测结果
├── post_train_probe_lora.json  # LoRA 训练后探测结果
├── train_log_lora.json          # LoRA 训练日志（40 条记录 + 1 条汇总）
├── figures/
│   └── train_curves.png         # 训练曲线图（loss / reward / lr）
└── grpo_lora_ckpt/              # LoRA checkpoint 目录
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── training_args.bin
    └── README.md
```

---

## 1. `baseline_probe.json` — 训练前基线探测

基座模型 `Qwen2.5-0.5B-Instruct` 在 6 个算术难度等级上的零样本表现（n=50 题，k=8 采样）。

| 难度 | greedy 格式率 | greedy 严格准确率 | 采样严格准确率 | pass@8 | 有效梯度组占比 |
|------|:---:|:---:|:---:|:---:|:---:|
| L1 一位数加法 | 0.46 | 0.46 | 0.4975 | 0.98 | 0.96 |
| L2 两位数加减 | 0.44 | 0.44 | 0.345 | 0.80 | 0.80 |
| L3 三位数加减 | 0.62 | 0.54 | 0.4175 | 0.90 | 0.88 |
| L4 一位数乘法 | 0.50 | 0.50 | 0.505 | 0.98 | 0.96 |
| L5 两位数×一位数乘法 | 0.52 | 0.50 | 0.4175 | 0.86 | 0.86 |
| L6 两位数×两位数乘法 | 0.52 | 0.30 | 0.2525 | 0.62 | 0.60 |

**关键观察**：

- 格式遵从率低（0.44–0.62），模型经常不输出 `<answer>` 标签，直接给数字
- 简单任务（L1–L4）的 loose 准确率远高于 strict（如 L1: loose 0.94 vs strict 0.46），说明模型算对了但格式不对
- L6 两位数乘法最难，严格准确率仅 0.30，pass@8 也只有 0.62
- 有效梯度组占比（`informative_group_rate`）高（0.60–0.96），说明 GRPO 能获得足够的正负奖励信号来学习

---

## 2. `post_train_probe_lora.json` — LoRA 训练后探测

同一组 6 个难度等级、同样 n=50 / k=8 的探测结果。

| 难度 | greedy 格式率 | greedy 严格准确率 | 采样严格准确率 | pass@8 | 有效梯度组占比 |
|------|:---:|:---:|:---:|:---:|:---:|
| L1 一位数加法 | 1.00 | 1.00 | 0.9675 | 1.00 | 0.26 |
| L2 两位数加减 | 1.00 | 0.98 | 0.9575 | 1.00 | 0.20 |
| L3 三位数加减 | 1.00 | 0.94 | 0.89 | 0.96 | 0.40 |
| L4 一位数乘法 | 1.00 | 0.98 | 0.9625 | 1.00 | 0.20 |
| L5 两位数×一位数乘法 | 1.00 | 0.96 | 0.9175 | 0.98 | 0.30 |
| L6 两位数×两位数乘法 | 0.98 | 0.46 | 0.46 | 0.60 | 0.26 |

**训练前后对比**：

| 难度 | greedy 严格 acc 提升 | 采样严格 acc 提升 | pass@8 提升 |
|------|:---:|:---:|:---:|
| L1 | +0.54 (0.46→1.00) | +0.47 (0.4975→0.9675) | +0.02 |
| L2 | +0.54 (0.44→0.98) | +0.6125 (0.345→0.9575) | +0.20 |
| L3 | +0.40 (0.54→0.94) | +0.4725 (0.4175→0.89) | +0.06 |
| L4 | +0.48 (0.50→0.98) | +0.4575 (0.505→0.9625) | +0.02 |
| L5 | +0.46 (0.50→0.96) | +0.50 (0.4175→0.9175) | +0.12 |
| L6 | +0.16 (0.30→0.46) | +0.2075 (0.2525→0.46) | -0.02 |

**关键观察**：

- 格式遵从事后全部接近 1.0，GRPO 的 `reward_format` 成分成功教会了模型输出 `<answer>` 标签
- L1–L5 严格准确率大幅提升（+0.40 到 +0.61），说明训练有效
- L6 两位数乘法提升有限（+0.16），模型在小参数量 + LoRA rank=16 下难以学会复杂乘法
- 有效梯度组占比大幅下降（0.60–0.96 → 0.20–0.40），因为模型变强后大部分题都对了，组内缺乏正负对比信号，这是 GRPO 训练后期的自然现象

---

## 3. `train_log_lora.json` — LoRA 训练日志

共 40 条日志记录（step 5 到 step 200，每 5 步记录一次）+ 1 条训练汇总。

### 训练配置概要

| 项目 | 值 |
|------|-----|
| 基座模型 | `Qwen2.5-0.5B-Instruct` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| target_modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| 总步数 | 200 |
| 总耗时 | 2202.6s（约 36.7 分钟） |
| 吞吐 | 2.906 samples/s, 0.091 steps/s |
| 总 token 数 | 503,218 |
| 最终 train_loss | -1.33e-05 |

### 训练曲线关键指标

| 指标 | 初始 (step 5) | 中段 (step 100) | 末段 (step 200) | 趋势 |
|------|:---:|:---:|:---:|------|
| loss | -0.030 | 0.0 | 4.5e-09 | 趋近于 0（GRPO loss 可为负） |
| grad_norm | 2.047 | 0.0 | 0.0 | 大部分步为 0（有效梯度组少） |
| learning_rate | 1.96e-04 | 1.01e-04 | 1.0e-06 | cosine schedule 衰减 |
| reward_correct | 0.8375 | 0.925 | 0.95 | 逐步上升 |
| reward_format | 0.1725 | 0.20 | 0.20 | 快速收敛到上限 |
| total reward | 1.01 | 1.125 | 1.15 | 上升 |
| frac_reward_zero_std | 0.50 | 0.90 | 0.90 | 上升（信号变稀疏） |
| entropy | 0.078 | 0.008 | 0.0078 | 下降（策略变确定） |
| completion 长度 | 10.18 | 9.95 | 10.15 | 稳定 ~10 token |

**关键观察**：

- `grad_norm` 大部分步为 0.0，因为 `frac_reward_zero_std` 高（组内无奖励差异→无有效梯度），这是 GRPO 在小模型 + 简单任务上的典型现象
- `reward_correct` 从 0.84 上升到 0.95，`reward_format` 快速收敛到 0.20 上限，说明训练有效
- `entropy` 从 0.078 降到 0.0078，策略分布变确定（greedy 解码越来越接近采样解码）
- `clip_ratio` 全程为 0，说明策略更新幅度小，没有触发 PPO 裁剪
- `completion` 长度稳定在 ~10 token，说明模型学会了简洁输出 `<answer>N</answer>`

---

## 4. `figures/train_curves.png`

训练曲线可视化图，包含 loss、reward、learning_rate 等指标随 step 变化的折线图，数据来源于 `train_log_lora.json`。

---

## 5. `grpo_lora_ckpt/` — LoRA Checkpoint

训练产出的 PEFT adapter 权重，可直接通过 `peft.PeftModel.from_pretrained()` 加载到基座模型上做推理。

| 文件 | 说明 |
|------|------|
| `adapter_config.json` | LoRA 配置（r=16, alpha=32, target=q/k/v/o_proj） |
| `adapter_model.safetensors` | LoRA adapter 权重 |
| `chat_template.jinja` | 对话模板 |
| `tokenizer_config.json` | tokenizer 配置 |
| `tokenizer.json` | tokenizer 词表 |
| `training_args.bin` | 完整训练参数序列化 |
| `README.md` | PEFT 自动生成的模型卡 |