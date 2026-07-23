# vLLM 云 GPU 部署指南

将本项目部署到云 GPU 实例，完成 vLLM 大模型服务的部署与速度验证。

---

## 一、云 GPU 平台选择

| 平台 | 最低费用 | 推荐配置 | 适合人群 |
|------|---------|---------|---------|
| [AutoDL](https://www.autodl.com) | ≈2-4 元/小时 | RTX 4060 / A4000（8GB 显存） | 新手首选，镜像多 |
| [矩池云](https://www.matpool.com) | ≈3-5 元/小时 | T4 / 2080Ti | 备选 |

**推荐 AutoDL**，社区镜像多，按量计费，用多少扣多少。

---

## 二、AutoDL 操作步骤

### Step 1：注册 & 充值

1. 打开 [AutoDL](https://www.autodl.com)，手机号注册
2. 实名认证（上传身份证，通常 5 分钟通过）
3. 充值 **10 元**（足够玩 3-4 小时）

### Step 2：创建实例

1. 进入"算力市场"，选一个 **RTX 4060** 或 **A4000** 的机器（8GB 显存）
2. 镜像选择：在社区镜像搜 **`vllm`**，选一个带 `vLLM 0.9.x` + `PyTorch 2.x` 的
   - 如果没有，选 `PyTorch 2.3.0 + CUDA 12.1 + Python 3.10` 基础镜像，用 `setup_cloud.sh` 自行安装
3. 数据盘：默认 50GB 足够
4. 点击"立即创建"，等待 1~2 分钟

### Step 3：上传代码

实例启动后，有三种方式：

**方式 A：JupyterLab 上传（最简单）**
1. 在 AutoDL 控制台点"JupyterLab"
2. 在左侧文件树右键 → 上传 → 选中本地 `vllm_cloud_deploy` 文件夹
3. 上传到 `/root/autodl-tmp/`

**方式 B：scp 命令行上传**
```bash
# 在 AutoDL 实例详情页复制"SSH 登录指令"，然后在本地终端执行：
scp -rP <端口号> vllm_cloud_deploy/ root@<IP>:/root/autodl-tmp/
```

**方式 C：从 GitHub clone（如果已推送到 GitHub）**
```bash
git clone <your-repo-url> /root/autodl-tmp/vllm_cloud_deploy
```

### Step 4：配置环境

在 AutoDL 的终端（JupyterLab 内 Terminal 或 SSH）执行：

```bash
cd /root/autodl-tmp/vllm_cloud_deploy
bash setup_cloud.sh
```

> 如果用的是带 vLLM 的镜像，可以跳过 `setup_cloud.sh`，手动 `pip install openai jsonschema matplotlib httpx` 即可。

---

## 三、执行顺序

### 路线图

```
setup_cloud.sh  →  start_server.sh  →  demo_*.py（约束解码） →  bench_throughput.py（吞吐）
    5min               20s                  8-12min                     3-5min
```

### 第一步：启动 vLLM Server

```bash
cd /root/autodl-tmp/vllm_cloud_deploy/src
bash start_server.sh
```

看到 `Application startup complete` 后，server 在 **8000 端口**运行。

### 第二步：跑约束解码 Demo

**新开一个终端**，运行 5 个 demo（推荐先跑核心的）：

```bash
cd /root/autodl-tmp/vllm_cloud_deploy/src

# ★ 推荐必跑（最核心，约 8-10 分钟）
python demo_function_call.py

# 其余各约 1-2 分钟
python demo_guided_choice.py
python demo_guided_regex.py
python demo_guided_json.py
python demo_response_format.py
```

### 第三步：吞吐 Benchmark

**先停掉 server**（释放显存），再跑：

```bash
fuser -k 8000/tcp
python bench_throughput.py
```

这会产出：
- 终端表格：三路对比（串行 / batch / vLLM）的总耗时、QPS、tokens/s
- `outputs/throughput_comparison.png`：柱状图
- `outputs/throughput_results.json`：原始数据

---

## 四、什么结果要截图（作业用）

跑完以上脚本，以下内容值得截图放进作业报告：

| 序号 | 截图内容 | 说明 |
|------|---------|------|
| 1 | `start_server.sh` 启动后显示 `Application startup complete` | 证明服务部署成功 |
| 2 | `curl http://localhost:8000/v1/models` 的结果 | 证明 API 可调用 |
| 3 | `demo_function_call.py` 的汇总表格 | 裸 prompt vs response_format vs guided_json 对比 |
| 4 | `demo_function_call.py` 的失败案例 | 展示 guided_json 如何修正错误 |
| 5 | `bench_throughput.py` 的**结果汇总表格** | vLLM 相对 transformers 的加速倍率 |
| 6 | `outputs/throughput_comparison.png` 柱状图 | 可视化加速效果 |

### 截图快速命令

```bash
# 查看 outputs 目录
ls -la outputs/

# 查看已产出文件
cat outputs/throughput_results.json
cat outputs/function_call_results.json | head -80
```

---

## 五、下载结果到本地

### 方式 A：JupyterLab 打包下载
1. 在 JupyterLab 文件树中右键 `outputs/` 文件夹 → 下载

### 方式 B：scp
```bash
scp -rP <端口号> root@<IP>:/root/autodl-tmp/vllm_cloud_deploy/outputs/ ./
```

### 方式 C：AutoDL 文件管理
控制台 → 实例 → 文件管理 → 勾选 → 下载

---

## 六、重要提示

### 务必关实例！
跑完记得**关机或销毁实例**，否则持续扣费：
- **关机**（推荐）：只收少量磁盘费（约 0.1 元/天），下次可继续用
- **销毁**：彻底清除，数据丢失

在 AutoDL 控制台，点"关机"即可。

### 模型下载提示
首次启动 `start_server.sh` 时，vLLM 会自动从 HuggingFace 下载 Qwen2-0.5B-Instruct（约 1GB）。
如果下载慢，设 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 如果显存不足
降低 `start_server.sh` 中的参数：
```bash
GPU_MEM_UTIL=0.4          # 从 0.6 降到 0.4
MAX_MODEL_LEN=1024         # 从 2048 降到 1024
```

---

## 七、预计花费

| 项目 | 费用 |
|------|------|
| RTX 4060 / A4000 实例运行 1 小时 | ≈ 2-4 元 |
| 磁盘费（关机状态） | ≈ 0.1 元/天 |
| **总计（跑完整个流程）** | **约 3-5 元** |

---

## 八、问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `Connection refused` | server 没启动 | 检查 `start_server.sh` 是否在运行 |
| `No CUDA devices` | 镜像不带 CUDA | 重选带 PyTorch+cu 的镜像 |
| 模型下载失败 | 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com` |
| OOM 显存溢出 | 显存不够 | 降低 `GPU_MEM_UTIL` 和 `MAX_MODEL_LEN` |
| `ModuleNotFoundError: vllm` | 依赖未装 | 运行 `bash setup_cloud.sh` |
