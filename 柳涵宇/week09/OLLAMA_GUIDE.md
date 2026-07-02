# OLLAMA_GUIDE.md — 用 Ollama 部署本项目模型

这份项目原本用 vLLM 启动 OpenAI 兼容服务，模型路径写在 `src/start_server.sh` 和 `src/bench_throughput.py` 里。  
如果改用 Ollama，不再需要本地 `MODEL_PATH`，而是通过 Ollama 的模型名调用，例如 `qwen2:0.5b`。

## 一、安装与启动

### 1.1 Windows 安装 Ollama

到官方页面安装 Windows 版 Ollama：

https://ollama.com/download

安装完成后，重新打开 PowerShell，验证：

```powershell
ollama --version
```

如果命令不存在，重启终端或重启电脑。

### 1.2 拉取 Qwen2 0.5B 模型

```powershell
ollama pull qwen2:0.5b
```

验证模型列表：

```powershell
ollama list
```

### 1.3 启动 Ollama 服务

Windows 版 Ollama 通常会自动在后台启动，默认地址：

```text
http://localhost:11434
```

验证：

```powershell
curl http://localhost:11434/api/tags
```

如果没有服务，手动运行：

```powershell
ollama serve
```

## 二、运行 Ollama Demo

进入项目目录：

```powershell
cd "D:\BaiduNetdiskDownload\week9大模型应用补充知识\week9 大模型应用补充知识\week9 大模型应用补充知识\vllm_deployment\src"
python demo_ollama_json.py
```

如果在 WSL2 里运行：

```bash
cd "/mnt/d/BaiduNetdiskDownload/week9大模型应用补充知识/week9 大模型应用补充知识/week9 大模型应用补充知识/vllm_deployment/src"
python3 demo_ollama_json.py
```

## 三、和 vLLM 版本的区别

Ollama 路线更适合本地快速体验：

- 不需要手写 HuggingFace 模型路径
- 不需要启动 vLLM server
- CPU 也能跑小模型，只是速度较慢
- 默认端口是 `11434`

但 Ollama 和 vLLM 的约束解码接口不同：

| 能力 | vLLM | Ollama |
|------|------|--------|
| OpenAI 兼容接口 | 支持 | 部分支持 |
| `guided_choice` | 支持 | 不支持 |
| `guided_regex` | 支持 | 不支持 |
| `guided_json` | 支持 | 不支持 vLLM 这个参数 |
| JSON 输出 | `response_format` / `guided_json` | `format: "json"` 或 JSON Schema |
| 吞吐优化 benchmark | 适合 | 不适合直接对比 |

所以本项目的原始 `demo_guided_*.py` 仍然是 vLLM 教学内容；Ollama 版本用 `demo_ollama_json.py` 演示结构化输出。

## 四、常见问题

### Q1：`ollama` 命令不存在

说明还没安装 Ollama，或安装后终端没有刷新 PATH。重启 PowerShell 或电脑。

### Q2：`connection refused`

Ollama 服务没启动。运行：

```powershell
ollama serve
```

### Q3：模型不存在

运行：

```powershell
ollama pull qwen2:0.5b
```

### Q4：想换模型

修改 `src/demo_ollama_json.py` 里的：

```python
MODEL = "qwen2:0.5b"
```

例如换成：

```python
MODEL = "qwen2.5:0.5b"
```

然后先拉取：

```powershell
ollama pull qwen2.5:0.5b
```
