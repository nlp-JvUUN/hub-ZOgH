# week09 — 部署 vLLM 大模型服务，验证速度提升

部署一个 vLLM（Qwen2-0.5B-Instruct）OpenAI 兼容服务，并用三路对照实验量化
**串行推理 / 批处理推理 / vLLM 批处理**的吞吐差距，验证 vLLM 的加速效果；
同时通过 `guided_json` 约束解码演示"服务 + 结构化输出"的最小应用闭环。

> **⚠️ 当前提交状态**：代码已就绪；`outputs/` 内为**参考数据**（老师标准数据，
> RTX 4060 / WSL2，见 `throughput_results.json` 的 `source` 字段），**非本机实测**。
> 本机（或云 GPU）可跑后，按第八节"实测清单"替换为实测数据即可。
> Mac 环境部署路线见 `Mac部署方案.md`。

## 目录结构

```
week09/
├── start_server.sh       # 部署：一键启动 vLLM OpenAI 兼容服务（端口 8000）
├── bench_throughput.py   # 验证：三路吞吐对比（串行 / batch=8 / vLLM）
├── demo_service.py       # 应用：通过服务 API 做对话 + guided_json 意图抽取
├── requirements.txt      # 依赖（版本兼容矩阵见下）
└── outputs/              # 运行后生成：throughput_results.json + throughput_comparison.png
```

## 环境准备

前置：WSL2 Ubuntu 22.04 + NVIDIA 驱动 566.x（CUDA 12.7），详见
`USAGE_GUIDE.md`。**关键版本矩阵**（装错会 `torch.cuda.is_available()=False`）：

| 组件 | 版本 | 原因 |
|------|------|------|
| vLLM | 0.9.2 | 0.20+ 要 CUDA 13（驱动 580+），笔记本驱动 566.x 不兼容 |
| torch | 2.7.0（cu126） | 与 vLLM 0.9.2 匹配 |
| transformers | 4.52.4 | 5.x 与 vLLM 0.9.2 冲突（aimv2 报错） |

```bash
source ~/vllm_env/bin/activate
cd <week09 目录>
pip install -r requirements.txt
python -c "import vllm, torch; print(vllm.__version__, torch.cuda.is_available())"
# 期望：0.9.2 True
```

## 使用步骤（完整闭环）

```bash
# 1. 部署：启动 vLLM 服务（等 "Application startup complete"，约 15~20 秒）
bash start_server.sh

# 2. 验证服务可用（新开终端）
curl http://localhost:8000/v1/models

# 3. 应用：通过服务调用模型（对话 + guided_json 意图抽取）
python demo_service.py

# 4. 验证速度：先停 server 释放显存（否则 transformers + vLLM 双模型 OOM）
fuser -k 8000/tcp
python bench_throughput.py            # 50 prompts × 100 tokens
# 快速冒烟：python bench_throughput.py --n 10 --max-tokens 32

# 5. 跑完可重启服务继续玩：bash start_server.sh
```

## 预期结果（参考：老师标准数据，RTX 4060 8GB / Qwen2-0.5B）

| 模式 | 50 请求总耗时 | QPS | tokens/s | 相对 vLLM |
|------|--------------|-----|----------|-----------|
| [A] transformers 串行 | ~61s | ~0.8 | ~60 | 0.017× |
| [B] transformers batch=8 | ~13s | ~3.9 | ~290 | 0.080× |
| [C] vLLM 批处理 | **~1s** | **~48** | **~3400** | **1.00×** |

**结论**：vLLM 相对串行约 **59×**，相对手写 batch 约 **12.5×**。
提速 = 批处理收益（~5×）× PagedAttention + continuous batching 收益（~12×）的乘积：
- 手写 batch 要 padding 到最长 prompt，浪费 60~80% 显存；
- PagedAttention 把 KV cache 按 block 分配，碎片 <4%，同显存容纳更大 batch；
- continuous batching 让短请求完成立即补新请求，GPU 利用率从 ~20% 拉到满载。

（不同设备数值会浮动，加速趋势一致；**本机实测以 `outputs/throughput_results.json` 为准**。）

## 实验要点

1. **控制变量**：同一模型、同一批 prompts、相同 max_new_tokens、temperature=0；
2. **指标口径**：三路 gen_tokens 不完全相等，结论围绕 **tokens/s（TPS）和倍率**讲，比 QPS 更公平；
3. **顺序约束**：bench 前必须先停 server（显存只有 8GB）；
4. **约束解码价值**：`response_format` 只保证 JSON 语法，`guided_json` 通过 FSM 屏蔽非法 token，
   保证字段名/枚举/正则/必填全部合法——这就是 Agent 工具调用可靠性的基础设施。

## 八、实测清单（当前为参考数据，可跑后按此替换）

```bash
# 1. 跑出本机实测（任选：WSL2/云 GPU/Colab/vllm-metal，详见 Mac部署方案.md）
bash start_server.sh && python demo_service.py
fuser -k 8000/tcp && python bench_throughput.py     # 重新生成 outputs/ 两个文件

# 2. 确认 JSON 里是实测数字（应显示为你的设备，且 source 字段已无"参考"字样）
# 3. 更新上方"预期结果"表为本机数字（老师数据可保留为参考列）
# 4. 推送（outputs/ 被 .gitignore 忽略，需强制添加）
git add -f week09/outputs/throughput_results.json week09/outputs/throughput_comparison.png
git commit -m "week09: vLLM 部署与吞吐对比实测" && git push
```

## 常见问题

| 问题 | 解法 |
|------|------|
| `torch.cuda.is_available()` 返回 False | 版本矩阵装错（装了 CUDA 13 的 torch），按上表降级 |
| server 报 `No available memory for the cache blocks` | `GPU_MEM_UTIL=0.4 bash start_server.sh` |
| demo 报 `Connection refused` | 服务没启动，先跑 `bash start_server.sh` |
| bench 显存溢出 | 忘了停 server：`fuser -k 8000/tcp` |
| 图表中文变方块 | 图内标签已用英文（DejaVu Sans 无中文字形） |
