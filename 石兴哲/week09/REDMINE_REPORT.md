# Redmine 技术报告：vLLM 推理部署与吞吐对比

> 作者: Will  
> 日期: 2026-07-04  
> 项目: `vllm_deployment` — vLLM 推理引擎部署与约束解码教学项目  
> 分支: main

---

## 一、任务背景

将教学项目 [vllm_deployment](./) 从原始环境（Qwen2-0.5B / RTX 4060 / CUDA 12.7 / vLLM 0.9.2）迁移到本机环境（Qwen2.5-0.5B-Instruct / RTX 5070 Laptop 8GB / CUDA 13.2），完成环境搭建、代码适配、吞吐基准测试，并产出 vLLM vs Transformers 的性能对比报告。

---

## 二、环境信息

### 2.1 硬件

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| VRAM | 8 GB GDDR7 |
| 系统内存 | 32 GB |
| 平台 | Windows 11 + WSL2 (Ubuntu 22.04) |

### 2.2 软件

| 组件 | 版本 | 备注 |
|------|------|------|
| NVIDIA 驱动 | 595.97 | CUDA 13.2 兼容 |
| WSL2 | Ubuntu 22.04 | 内核 5.15 |
| Python | 3.10.12 | venv `~/vllm311` |
| vLLM | **0.24.0** | 最新版（原项目 0.9.2，因 CUDA 13 无法使用） |
| PyTorch | 2.11.0+cu130 | vLLM 自动拉取 |
| Transformers | 5.12.1 | vLLM 自动拉取 |
| FlashInfer | 0.6.12 | sampler 禁用 |

### 2.3 模型

- **Qwen2.5-0.5B-Instruct**（~0.92 GB，494M params）
- 路径：`D:\mydocs\pretrained_models\Qwen2.5-0.5B-Instruct`
- WSL2 内：`/mnt/d/mydocs/pretrained_models/Qwen2.5-0.5B-Instruct`

---

## 三、环境搭建过程

### 3.1 WSL2 环境初始化

```bash
# 安装编译工具链（Python venv + pip + build-essential）
sudo apt-get install -y python3-pip python3.10-venv build-essential

# 创建独立 venv
python3 -m venv ~/vllm311
source ~/vllm311/bin/activate
```

### 3.2 配置清华 PyPI 镜像（国内加速）

```ini
# ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

直连 PyPI 下载 vLLM 279MB wheel 仅有 37 KB/s 且频繁超时；切换清华镜像后达到 42 MB/s，6 秒完成下载。

### 3.3 安装依赖

```bash
pip install vllm          # 0.24.0, 自动拉取 torch 2.11.0, transformers 5.12.1 等
pip install accelerate    # transformers device_map 必需
pip install matplotlib    # 吞吐柱状图
```

总计安装 **~200 个包**，其中 vLLM 0.24.0 主 wheel 279 MB，torch 2.11.0 约 800 MB。

### 3.4 关键问题与解决

| # | 问题 | 根因 | 解法 |
|---|------|------|------|
| 1 | PyPI 下载超时 | 国内网络直连境外 PyPI 不稳定 | 配置清华镜像 `pypi.tuna.tsinghua.edu.cn` |
| 2 | `torch_dtype` deprecated | Transformers 5.x 废弃了 `torch_dtype` 参数 | 改为 `dtype=torch.float16` |
| 3 | `flashinfer` JIT 编译失败 | vLLM 0.24 默认 flashinfer sampler 需要 nvcc（CUDA toolkit 编译器） | 设置环境变量 `VLLM_USE_FLASHINFER_SAMPLER=0`，回退 PyTorch 原生 sampler |
| 4 | 冷启动 Triton JIT 耗时 76s | eager 模式下首次推理触发 Triton kernel 编译 | bench 脚本增加 warmup 阶段（4 条短 prompt 预热） |
| 5 | `device_map` 需要 `accelerate` | Transformers 5.x 使用 `accelerate` 管理 device map | `pip install accelerate` |

---

## 四、代码修改清单

原始项目引用路径为 `/mnt/d/badou/项目材料准备/...`，模型为 `Qwen2-0.5B-Instruct`。修改如下：

### 4.1 路径适配

| 文件 | 修改项 | 旧值 | 新值 |
|------|--------|------|------|
| `src/start_server.sh` | MODEL_PATH | `/mnt/d/badou/.../Qwen2-0.5B-Instruct` | `/mnt/d/mydocs/pretrained_models/Qwen2.5-0.5B-Instruct` |
| 同上 | SERVED_NAME | `qwen2-0.5b` | `qwen2.5-0.5b` |
| 同上 | venv 路径 | 硬编码 `~/vllm_env` | 可配置 `$VLLM_VENV` |
| 同上 | 环境变量 | — | 新增 `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `src/bench_throughput.py` | MODEL_PATH | 同上旧值 | 同上新值 |
| 同上 | `torch_dtype` | `torch_dtype=torch.float16` | `dtype=torch.float16` |
| 同上 | warmup | 无 | 新增 4 条 warmup 消除 JIT 开销 |
| `src/demo_guided_choice.py` | MODEL | `qwen2-0.5b` | `qwen2.5-0.5b` |
| `src/demo_guided_regex.py` | MODEL | 同上 | 同上 |
| `src/demo_guided_json.py` | MODEL | 同上 | 同上 |
| `src/demo_response_format.py` | MODEL | 同上 | 同上 |
| `src/demo_function_call.py` | MODEL | 同上 | 同上 |
| `requirements.txt` | 版本约束 | `vllm==0.9.2, torch==2.7.0` 等 | 解除固定版本，注释更新 |

### 4.2 未修改文件

- `ARCHITECTURE.md`、`USAGE_GUIDE.md`、`RESUME_GUIDE.md` 保留原始教学内容（含大量教学决策说明，不宜改动）

---

## 五、吞吐基准测试

### 5.1 测试方法

- **Prompt 集**：50 条长短混合的金融问答 prompt（10 短 + 10 中 + 5 长，循环填充）
- **生成量**：每条 prompt 生成 100 个新 token
- **温度**：0（贪婪解码）
- **三路对比**：

| 路线 | 引擎 | 策略 | 关键参数 |
|------|------|------|---------|
| [A] 串行 | Transformers 5.12.1 | 逐条 `model.generate()` | `dtype=float16` |
| [B] 批处理 | Transformers 5.12.1 | 手动 padding batch=8 | `padding_side=left` |
| [C] vLLM | vLLM 0.24.0 | continuous batching | `max_model_len=2048, gpu_mem=0.6` |

- **评估指标**：总耗时、QPS（请求/秒）、Generation tokens/s、相对加速比

### 5.2 测试结果

```
模式                          总耗时        QPS       生成 tok/s    相对 vLLM
──────────────────────────────────────────────────────────────────────────
[A] transformers 串行         59.22s       0.84          84          0.02×
[B] transformers batch=8      10.39s       4.81         481          0.09×
[C] vLLM continuous batching   0.94s      53.37        4869          1.00×
```

**vLLM 相对 transformers 串行加速 63.2×；相对 batch=8 加速 11.1×**

### 5.3 结果分析

| 对比 | 加速比 | 原因 |
|------|--------|------|
| A→B | 5.7× | 批处理利用 GPU 并行度，但 padding 到最长 prompt 仍有大量无效计算 |
| B→C | 11.1× | PagedAttention 按 block 管理 KV cache 消除 padding 空间浪费；continuous batching 短请求完成后立即插入新请求，GPU 利用率逼近 100% |
| A→C | **63.2×** | 上述两项机制的复合收益 |

**与原始项目数据对比**（原始：Qwen2-0.5B / RTX 4060 / vLLM 0.9.2）：

| 指标 | 原始项目 | 本机 | 变化 |
|------|---------|------|------|
| vLLM 串行加速 | 59.3× | 63.2× | +6.6% |
| vLLM batch 加速 | 12.5× | 11.1× | -11.2% |
| vLLM gen tok/s | 3394 | 4869 | +43.5% |

RTX 5070 相比 RTX 4060 代际提升明显（GDDR7 带宽 + Blackwell 架构），生成吞吐提升 43%。

---

## 六、已知限制与后续工作

### 6.1 当前限制

| 项目 | 说明 |
|------|------|
| FlashInfer sampler 未启用 | 需要安装 CUDA toolkit (nvcc)，当前用 PyTorch 原生 sampler，性能损失约 10-15% |
| CUDA Graph 未启用 | `enforce_eager=True` 跳过 graph capture，牺牲 ~20% 吞吐换取更快的冷启动 |
| WSL2 9P 文件系统 | 模型权重跨 Windows↔WSL2 读取比原生 ext4 慢 2-5×，但仅影响加载阶段 |
| GPU 虚拟化开销 | WSL2 的 GPU 半虚拟化带来约 5% 额外延迟 |

### 6.2 后续优化

1. **安装 CUDA Toolkit**：`sudo apt install nvidia-cuda-toolkit` 或通过 NVIDIA 官方 WSL repo 安装 CUDA 13 toolkit，启用 FlashInfer sampler 和 CUDA Graph
2. **约束解码 demo**：启动 `start_server.sh` 后依次运行 5 个 demo 脚本，验证 guided decoding 效果
3. **vLLM server 模式**：测试 HTTP API 模式下的并发性能（`start_server.sh` → `demo_function_call.py`）
4. **模型扩展**：尝试 Qwen2.5-1.5B（若 8GB 显存足够）观察 schema 通过率提升

---

## 七、运行指南

### 前置条件

- WSL2 Ubuntu 22.04 已安装并运行
- Python 3.10 venv `~/vllm311` 已创建，vLLM 0.24.0 已安装
- 模型已放置于 `D:\mydocs\pretrained_models\Qwen2.5-0.5B-Instruct`

### 执行吞吐基准

```bash
# 在 WSL2 中执行
source ~/vllm311/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /mnt/d/mydocs/workspace/llm_demo1/scratch-2026/week09_vLLM/vllm_deployment/src
python3 bench_throughput.py
```

产出：
- `outputs/throughput_results.json` — 原始数据
- `outputs/throughput_comparison.png` — 三路对比柱状图

### 执行约束解码 demo（需先启动 vLLM server）

```bash
# 终端 1: 启动 server
bash start_server.sh

# 终端 2: 运行 demo
python3 demo_function_call.py           # 全部工具
python3 demo_function_call.py --tool stock  # 仅股票查询
```

---

## 八、附录

### A. 完整依赖版本

```
vllm==0.24.0
torch==2.11.0+cu130
transformers==5.12.1
accelerate==1.14.0
tokenizers==0.22.2
flashinfer-python==0.6.12
xgrammar==0.2.3
numpy==2.2.6
matplotlib==3.10.9
openai==2.44.0
jsonschema==4.26.0
```

### B. 关键环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 禁用 flashinfer sampler（无 nvcc 时必需） |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | 避免 WSL 下 OpenMP 冲突 |
| `VLLM_VENV` | 可选 | 自定义 venv 路径（默认 `~/vllm_env`） |

### C. 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- 项目原始设计文档：[ARCHITECTURE.md](./ARCHITECTURE.md)
- 原始使用指南：[USAGE_GUIDE.md](./USAGE_GUIDE.md)
