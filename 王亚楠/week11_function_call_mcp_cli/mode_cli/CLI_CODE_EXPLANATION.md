# mode_cli 代码逐行详解

本文对以下两个文件进行逐句注释讲解：

- `mode_cli/cli/main.py` — fincli 命令行工具本身
- `mode_cli/run_cli.py` — LLM 通过 Function Call 调用 CLI 的两种形态

---

## 一、`mode_cli/cli/main.py` — fincli 命令行工具

这个文件是整个项目的"工具实现层"，它把后端能力封装成一条可在终端直接运行的命令，和 MCP、Function Call 等协议完全无关。

### 1.1 模块文档字符串

```python
"""
main.py — fincli：A股年报检索 + 天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。
"""
```

**讲解**：这行注释点明了这个文件的核心价值——让 Python 脚本变成一条**真实命令**。对比：
- 不好的体验：`python ~/project/mode_cli/cli/main.py search --query "营收"` —— 又长又丑
- 好的体验：`fincli search --query "营收"` —— 简短优雅，和 `git`/`ls` 一样自然

```python
"""
通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli list-companies
  fincli search --query "营收和净利润" --stock-code 300750 --year 2023 --top-k 3
  fincli weather --city 宁德
"""
```

**讲解**：这里给出了具体的用法示例。关键在 `pyproject.toml` 中的这段配置：

```toml
[project.scripts]
fincli = "mode_cli.cli.main:main"
```

这行配置告诉 pip：安装时创建一个名为 `fincli` 的可执行文件，它调用 `mode_cli.cli.main` 模块中的 `main` 函数。原理是 pip 会在 `PATH` 下生成一个 wrapper 脚本。

```python
"""
不想安装也可直接跑：
  python mode_cli/cli/main.py search --query "营收" --stock-code 300750 --year 2023
  python -m mode_cli.cli.main weather --city 宁德
"""
```

**讲解**：提供了不安装的降级方案，方便开发和临时使用。

```python
"""
教学点：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本——跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含多个子命令（search/list-companies/weather），对应 git 的子命令设计
"""
```

**讲解**：这里点明了三个核心教学点：
1. **分层设计**：CLI 只负责"执行"，不关心谁在调用它（人 or LLM）。这和 MCP Server 是平级的——两者都调用同一个 `src/` 后端。
2. **标准发布**：`pyproject.toml` + `[project.scripts]` 是 Python 社区推荐的标准做法（替代旧的 `setup.py` + `entry_points`）。
3. **子命令模式**：借鉴 `git commit`/`git push` 的设计，用户只需记住一个命令名 `fincli`。

```python
"""
依赖：
  pip install faiss-cpu numpy openai httpx
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""
```

**讲解**：这里只声明了 Embedding 依赖（DashScope），因为这个 CLI 脚本自己不做 LLM 调用——它只调用后端的检索和天气功能。

---

### 1.2 导入部分

```python
import argparse  # 标准库：命令行参数解析
import sys       # 标准库：系统相关操作，这里用于sys.path
from pathlib import Path  # 标准库：处理文件路径，跨平台
```

```python
# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

**讲解**：这是 Python 项目中常见的路径处理技巧。逐层拆解 `Path(__file__).parent.parent.parent`：

- `__file__` = `.../function_call_mcp_cli/mode_cli/cli/main.py`
- `.parent` = `.../function_call_mcp_cli/mode_cli/cli/`
- `.parent.parent` = `.../function_call_mcp_cli/mode_cli/`
- `.parent.parent.parent` = `.../function_call_mcp_cli/`（项目根）

`sys.path.insert(0, ...)` 把项目根插到 Python 搜索路径的最前面，确保后续 `from src.xxx import ...` 能找到模块。

> **为什么用 `insert(0, ...)` 而不是 `append`？**
> `insert(0, ...)` 插到列表最前面，优先级最高。如果系统里刚好装了一个也叫 `src` 的包，我们的优先匹配。这是一种防御性编程。

```python
from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402
```

**讲解**：
- `# noqa: E402` 告诉 linter（如 flake8）忽略 "module level import not at top of file" 警告。因为必须先 `sys.path.insert`，再 `import`，否则找不到模块。
- 导入的三个函数是所有上层协议（CLI / Function Call / MCP）共同调用的**后端实现**。

---

### 1.3 主函数 `main()`

```python
def main():
```

**讲解**：`main` 函数名对应 `pyproject.toml` 中的 `fincli = "mode_cli.cli.main:main"`。

```python
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
```

**讲解**：
- `ArgumentParser` 是 Python 标准库的命令行解析器，自动处理 `--help`、参数验证、类型转换等。
- `prog="fincli"` 设置程序名（显示在 `--help` 输出中）。
- `sub = parser.add_subparsers(dest="cmd", required=True)` 创建子命令系统。
  - `dest="cmd"` 表示用户输入的子命令名会被存到 `args.cmd` 属性中。
  - `required=True` 表示必须输入子命令（Python 3.7+ 才支持此参数）。

```python
    # fincli search ...
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含公司名/年份，用简短财务术语，如 '营收和净利润'）")
    p_search.add_argument("--stock-code", default=None, help="按公司过滤，如 300750")
    p_search.add_argument("--year", default=None, help="按年份过滤：2021/2022/2023")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")
```

**讲解**：定义 `search` 子命令，它接收 4 个可选参数：

| 参数 | 必填 | 类型 | 默认值 | 作用 |
|------|------|------|--------|------|
| `--query` | ✅ | str | — | 检索问题，这里特别提示不要含公司名/年份 |
| `--stock-code` | ❌ | str | `None` | 按股票代码过滤 |
| `--year` | ❌ | str | `None` | 按年份过滤 |
| `--top-k` | ❌ | int | `5` | 返回段落数量 |

- `type=int` 让 argparse 自动把用户输入的字符串转为整数，无需手动 `int("5")`。
- `default=None` 意味着不传这个参数时，它的值是 `None`，后端函数会跳过对应过滤条件。

```python
    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录的公司")
```

**讲解**：`list-companies` 子命令没有任何参数，只是一个简单的 list 操作。连 `p_` 变量名都没赋值——因为这个 parser 不需要额外添加参数。

```python
    # fincli weather ...
    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")
```

**讲解**：`weather` 子命令只需一个 `--city` 参数。

```python
    args = parser.parse_args()
```

**讲解**：`parse_args()` 解析 `sys.argv`（命令行参数列表）。如果用户输入不合法（比如 `search` 没传 `--query`），argparse 自动打印错误信息并退出。解析结果是一个 Namespace 对象 `args`，可以用 `args.cmd`、`args.query` 等方式访问。

```python
    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "weather":
        print(get_weather(args.city))
```

**讲解**：根据解析出的子命令名 `args.cmd`，调用对应的后端函数，并把结果 `print` 到 stdout。

> **重要设计**：这里用 `print()` 输出到 stdout，意味着调用方（人、shell 脚本、LLM）都可以通过捕获 stdout 获取结果。这正是 CLI 的哲学——"一个工具做好一件事，通过文本流连接"。

```python
if __name__ == "__main__":
    main()
```

**讲解**：Python 标准入口守卫。当文件被直接执行时（`python main.py` 或 `fincli`）运行 `main()`；当文件被其他模块 import 时不运行。

---

## 二、`mode_cli/run_cli.py` — LLM 通过 Function Call 调用 CLI

这个文件展示了**让 LLM 驱动命令行工具**的两种形态。Function Call 在这里扮演的是"意图生成层"——LLM 理解用户问题，决定调用哪个工具、传什么参数，然后由 host 执行。

### 2.1 模块文档字符串

```python
"""
run_cli.py — 方式三：CLI（命令行即工具），两种形态
"""
```

**讲解**：之前项目已经展示了两种让 LLM 调用工具的方式：
- **方式一（Function Call）**：LLM 直接调用 Python 函数
- **方式二（MCP）**：通过 MCP 协议接入工具

这里是**方式三（CLI）**：把命令行工具作为 LLM 可调用的工具。

```python
"""
教学重点：
  1. 形态 A（具名 run_cli）：LLM 调一个 run_cli(command, args) 工具，command 是白名单 enum，
     host 拼出子命令执行。安全可控，但每加一个命令要改代码
  2. 形态 B（通用 run_bash）：LLM 自己拼完整 shell 命令，host 在沙箱里执行。
     最灵活、最危险——教学重点是沙箱设计（白名单/黑名单/超时/工作目录锁定）
  3. 与前两方式对比：CLI 是"工具实现层"，Function Call 是"意图生成层"，MCP 是"协议接入层"
     三者不互斥：run_cli/run_bash 本身也是用 Function Call 触发的
"""
```

**讲解**：这是本文件的精华——对比两种形态：

| 维度 | 形态 A（named） | 形态 B（bash） |
|------|----------------|----------------|
| LLM 输出 | `run_cli(command="rag_search", args={...})` | `run_bash(command="fincli search --query '营收'")` |
| LLM 自由度 | 受限（只能选白名单命令） | 可以拼任意 shell 命令 |
| 安全性 | 高（host 组装命令） | 低（需沙箱限制） |
| 扩展性 | 差（新增命令要改代码） | 好（新增工具无需改代码） |
| 教学价值 | "安全可控"的范例 | "沙箱设计"的范例 |

```python
"""
使用方式：
  # 先把 fincli 装成 PATH 上的真实命令（一次即可）
  pip install -e .
"""
```

**讲解**：`pip install -e .` 的 `-e` 表示 editable（开发模式），意思是安装后修改源码立即生效，不需重新安装。适合开发阶段。

```python
"""
  # 形态 A（具名，默认）
  python mode_cli/run_cli.py --mode named --question "宁德时代2023年营收和净利润？"
  # 形态 B（通用 bash）
  python mode_cli/run_cli.py --mode bash --question "宁德时代2023年营收和净利润？"
  # 内置示例
  python mode_cli/run_cli.py --mode named --demo
"""
```

---

### 2.2 导入部分

```python
import json       # 解析 LLM 返回的 tool_call.arguments (JSON 字符串)
import os         # 读取环境变量（API Key）
import re         # 正则匹配——沙箱的危险命令黑白名单
import shlex      # shell 命令词法解析——安全地拆分命令行字符串
import shutil     # shutil.which() —— 检查命令是否在 PATH 中
import subprocess # 子进程管理——执行 CLI 命令
import sys        # stderr 输出、sys.path 操作
import time       # 计时（统计 LLM 调用耗时）
from pathlib import Path  # 文件路径操作
```

```python
from openai import OpenAI  # OpenAI 兼容客户端——这里用来调 DeepSeek 和 DashScope
```

**讲解**：DeepSeek 和 DashScope（千问）都兼容 OpenAI 的 API 格式，所以用同一个 `OpenAI` 客户端就能调。

```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # 确保能 import 项目中的模块
```

```python
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
CLI_DIR = Path(__file__).parent / "cli"  # mode_cli/cli 目录
PY = sys.executable                       # 当前 Python 解释器路径
```

**讲解**：
- `BASE_DIR` = `function_call_mcp_cli/`
- `CLI_DIR` = `function_call_mcp_cli/mode_cli/cli/`
- `PY` = 当前 Python 可执行文件路径（如 `/opt/miniconda3/bin/python`），用于子进程调用

### 2.3 fincli 命令定位

```python
# fincli 真实命令路径：优先用 pip install -e . 注册到 PATH 的 fincli；
# 没装就退回 python mode_cli/cli/main.py（保证不安装也能跑，只是命令不"漂亮"）
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"
```

**讲解**：这段代码做了一个优雅的降级设计：

1. `shutil.which("fincli")` 在 `PATH` 中查找 `fincli` 命令
2. 找到了 → `FINCLI_ARGV = ["fincli"]`，直接用命令名
3. 没找到 → `FINCLI_ARGV = [python路径, "main.py路径"]`，退回脚本方式
4. `FINCLI_LABEL` 是给人看的标签（日志/提示用）

这样设计的好处：**功能不受安装状态影响**，装了就好看，没装也能跑。

### 2.4 LLM 配置

```python
PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}
```

**讲解**：两个 LLM Provider 的配置字典。因为都兼容 OpenAI API 格式，所以只需不同的 `api_key` 和 `base_url`。

```python
def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]
```

**讲解**：
- 从 `PROVIDERS` 取配置
- 检查 API Key 是否存在，不存在则打印错误并退出（`sys.exit(1)` 表示异常退出，退出码为 1）
- 返回 `(OpenAI客户端, 模型名)` 元组。注意 `OpenAI` 客户端的 `base_url` 指向 DeepSeek/DashScope 而非 OpenAI 官方服务器

### 2.5 形态 A：具名 `run_cli`（安全可控）

```python
# ── 形态 A：具名 run_cli ───────────────────────────────────────────────────
# 白名单 enum 限定可执行命令集——这是"安全"的来源：模型只能调预先批准的命令。
# 底层统一走 fincli（一条真实命令），而非 python xxx.py，更接近真实 CLI 工具形态。
```

**讲解**：形态 A 的核心设计思想：**LLM 只负责决策"调用哪个命令 + 传什么参数"，host 负责"拼装并执行真正的命令"**。LLM 永远接触不到 shell，安全风险趋近于零。

```python
# command 名 → 实际执行的 argv 模板（参数由 LLM 通过 args JSON 提供）
NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
    },
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}
```

**讲解**：白名单映射表。每个命令包含两部分：

- `argv`：固定的命令前缀（如 `["fincli", "search"]` 或 `["python", "main.py", "search"]`）
- `arg_map`：LLM 返回的 JSON key → CLI flag 的映射。例如 LLM 返回 `{"query": "营收和净利润"}`，host 会拼成 `--query "营收和净利润"`

关键设计：**`arg_map` 不是 1:1 透传，而是 host 控制的映射**。即使 LLM 被 prompt injection 攻击，返回了 `{"query": "'; rm -rf /'"}`，host 也只是把它当成 `--query` 的参数值传给 `subprocess.run(argv)`，不会执行任何 shell 命令。

对比：如果 LLM 直接拼 shell 字符串 `f"fincli search --query '{query}'"`，同样的注入 `'; rm -rf /'` 就会导致命令执行。形态 A 用 `argv` 列表（而非字符串 shell）完全避免了这个问题。

```python
def run_named(command: str, args: dict) -> str:
    """形态 A：按白名单拼出 argv，子进程执行，返回 stdout。"""
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"
```

**讲解**：首先查白名单。不在白名单的直接拒绝，返回错误信息给 LLM。这是安全的第一道防线。

```python
    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])
```

**讲解**：拼装命令行参数。
- `list(spec["argv"])` 创建副本，避免修改原始数据
- 遍历 `arg_map`，如果 LLM 传了对应的值（不为 `None`），就追加 `[flag, value]`
- 例如：`argv` 从 `["fincli", "search"]` 变成了 `["fincli", "search", "--query", "营收和净利润", "--stock-code", "300750", "--year", "2023"]`
- 注意这里没有 `shell=True`，所以参数值中的特殊字符（空格、分号等）不会触发 shell 解析

```python
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout
```

**讲解**：`subprocess.run` 的参数含义：

| 参数 | 值 | 含义 |
|------|-----|------|
| `capture_output=True` | 捕获 stdout 和 stderr | 不让子进程输出污染当前终端 |
| `text=True` | 输出为字符串 | 否则是 bytes |
| `timeout=30` | 超时 30 秒 | 防止死循环或无响应 |
| `cwd=str(BASE_DIR)` | 工作目录锁定为项目根 | 子进程无法访问其他目录 |
| `env={**os.environ}` | 继承当前环境变量 | 确保 API Key 等信息可用 |

- `TimeoutExpired`：子进程超时，返回错误信息
- `proc.returncode != 0`：子进程异常退出，返回最后 500 字符的 stderr
- 正常：返回 stdout 内容

### 2.6 形态 B：通用 `run_bash`（沙箱设计）

```python
# ── 形态 B：通用 run_bash（沙箱）──────────────────────────────────────────
# 模型自己拼 shell 命令字符串——最灵活也最危险，沙箱是教学重点。
```

**讲解**：形态 B 的核心：LLM 直接输出一条完整的 shell 命令字符串。这是最灵活的方式，但也最危险——所以设计了多层沙箱防线。

#### 2.6.1 危险命令黑名单

```python
# 危险命令黑名单（正则，命中即拒绝执行）
DANGEROUS_PATTERNS = [
    r"\brm\b",              # 删除文件
    r"\bdel\b",             # 删除（Windows）
    r"\brmdir\b",           # 删除目录
    r"\bdeltree\b",         # 删除目录树（Windows）
    r"\bformat\b",          # 格式化磁盘
    r"\bmkfs\b",            # 创建文件系统（Linux 格式化）
    r"\bdd\b",              # 磁盘写入（可破坏分区表）
    r"\bshutdown\b",        # 关机
    r"\breboot\b",          # 重启
    r"\bpoweroff\b",        # 断电
    r"[>;]\s*(?:rm|del|format)\b",  # 通过重定向/分号拼接的删除命令
    r"\bcurl\b.*\|\s*sh",   # curl 管道到 sh（远程代码执行）
    r"\bwget\b.*\|\s*sh",   # wget 管道到 sh（远程代码执行）
    r"\bsudo\b",             # 提权
    r"\bchmod\b.*-R",       # 递归修改权限
    r"\bchown\b.*-R",       # 递归修改所有者
    r"\bnc\b", r"\bnetcat\b",  # netcat——常用于反弹 shell
    r"/etc/passwd",           # 敏感文件
    r"/etc/shadow",           # 敏感文件
    r"\bTaskkill\b",          # 杀进程（Windows）
    r"\bStop-Process\b",      # 杀进程（Windows PowerShell）
]
```

**讲解**：黑名单用正则匹配命令字符串中的危险模式。`\b` 是单词边界，确保 `rm` 不会误匹配到 `format` 中的 `rm`。这是沙箱的第一层防线——**模式匹配**。

但黑名单有天然的局限性：
- 编码绕过（Base64、Hex 等）可以避开正则
- OOD 问题——总有你没想到的危险命令

所以还需要白名单和其他机制配合。

#### 2.6.2 命令头白名单

```python
# 命令白名单：只允许这些可执行文件作为命令头（其余拒绝）
# 形态 B 仍要危险可控：只放行 fincli（本项目工具）+ python + 几个只读命令
ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}
```

**讲解**：只允许特定的"命令头"——取命令行第一个 token 的文件名检查。这是沙箱的第二层防线——**只放行已知安全的可执行文件**。

例如：
- `fincli search --query "营收"` → head 是 `fincli` ✅ 通过
- `fincli; rm -rf /` → 虽然有分号，但 head 仍是 `fincli` ✅（黑名单会拦截 `rm`）
- `bash -c "echo hello"` → head 是 `bash` ❌ 不在白名单

#### 2.6.3 沙箱检查函数

```python
def sandbox_check(command: str) -> str | None:
    """返回 None 表示通过；返回字符串表示拒绝原因。"""
    # 第一层：黑名单正则匹配
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"

    # 第二层：命令头白名单
    try:
        tokens = shlex.split(command, posix=True)  # 安全拆分 shell 命令
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()  # 取第一个 token 的文件名（去掉路径前缀）
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None  # 全部通过
```

**讲解**：
- `shlex.split(command, posix=True)` 是安全的 shell 命令解析器，能正确处理引号、转义等。例如 `echo "hello world"` → `["echo", "hello world"]`
- `Path(tokens[0]).name` 提取文件名部分，防止用路径绕过。例如 `/usr/bin/rm` → `rm`
- 返回值设计为 `str | None`：`None` 表示安全，`str` 是拦截原因

#### 2.6.4 执行函数

```python
def run_bash(command: str) -> str:
    """形态 B：模型生成的 shell 命令，经沙箱检查后在锁定工作目录执行。"""
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        # shell=True 让模型可以用管道/重定向；工作目录锁在项目根；
        # 超时 15s 防止死循环；不继承会话的交互式特性
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out
```

**讲解**：形态 B 的沙箱多层防线总结：

| 防线 | 机制 | 代码位置 |
|------|------|----------|
| 第 1 层 | 正则黑名单 | `DANGEROUS_PATTERNS` + `sandbox_check()` |
| 第 2 层 | 命令头白名单 | `ALLOWED_HEADS` + `sandbox_check()` |
| 第 3 层 | 工作目录锁定 | `cwd=str(BASE_DIR)` — 子进程只能操作项目目录 |
| 第 4 层 | 超时限制 | `timeout=15` — 防止死循环或长时间占用 |
| 第 5 层 | 环境隔离 | `env={**os.environ}` — 创建新的环境字典副本 |

注意 `shell=True` 的必要性：形态 B 允许 LLM 使用管道（`|`）、重定向（`>`）等 shell 特性，所以必须 `shell=True`。但这也意味着即使有白名单，**shell injection 的残余风险仍然存在**（例如参数值中注入分号后的命令）。这就是为什么同时有黑名单+白名单+超时+目录锁定的多层次防护。

---

### 2.7 两种形态的 tools schema

```python
NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查 A 股年报（rag_search/list_companies）和天气（weather）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),  # ["rag_search", "rag_list_companies", "weather"]
                        "description": "...",
                    },
                    "args": {
                        "type": "object",
                        "description": "...",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
```

**讲解**：这是手动编写的 OpenAI Function Call tools schema。注意与之前讲的 `@mcp.tool()` 自动生成 JSON Schema 形成对比——这里 JSON Schema 是手写的。

关键设计：
- `"enum": list(NAMED_COMMANDS.keys())` 直接把白名单命令列表嵌入了 schema。LLM 看到的 schema 中 command 只能取这三个值，从源头限制了 LLM 的决策空间。
- `args` 是一个自由 JSON 对象，LLM 可以往里填任意参数，host 通过 `arg_map` 映射到 CLI flag。

```python
BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "可用工具 fincli（一条真实命令）："
                "fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3；"
                "fincli list-companies；"
                "fincli weather --city 宁德。"
                "危险命令（rm/del/format/sudo/curl|sh 等）会被拦截；只允许白名单可执行文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]
```

**讲解**：形态 B 的 schema 极其简单——只有一个 `command` 字段。**安全不靠 schema，全靠 host 沙箱**。

注意 description 中的几个教学细节：
1. 给出了 `fincli` 命令的具体用法示例（引导 LLM 写出正确的命令）
2. 明确告知了知识库范围和 query 写法规则
3. 警告了沙箱限制（危险命令会被拦截）

### 2.8 模式分发字典

```python
# 形态 → (schema, executor)
MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}
```

**讲解**：一个优雅的分发机制。将两种形态的 schema 和执行函数打包在一起，后续代码只需查字典即可获取对应模式的所有信息。

- `lambda args: run_named(args["command"], args.get("args", {}))` 是一个适配器函数：从 LLM 传回来的完整参数对象中，提取 `command` 和 `args` 子字段，分别传给 `run_named`。

### 2.9 LLM System Prompt

```python
SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手。通过 run_cli 工具调用预批准命令查 A 股年报与天气。"
    "回答年报问题前必须先 run_cli(command='rag_search', args={...}) 检索原文，只依据返回段落作答，不要编造。"
    "知识库仅含：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)，年份 2021-2023。"
    "rag_search 的 query 不要含公司名/年份（已由 stock_code/year 过滤），用简短术语如 '营收和净利润'。"
    "不在库内的公司请明确告知，不要臆测。本回合可一次调用多个工具。"
)
```

**讲解**：System Prompt 的教学要点：

1. **强制检索**："回答年报问题前必须先 run_cli(...) 检索原文"——防止 LLM 直接凭训练记忆瞎编
2. **知识边界**："知识库仅含 5 家公司，年份 2021-2023"——防止 LLM 查找不存在的公司
3. **query 优化规则**："不要含公司名/年份"——因为公司名和年份已通过 `stock_code`/`year` 精确过滤，混入 query 会稀释向量检索精度
4. **诚实原则**："不在库内的公司请明确告知"——防止 LLM 编造数据

```python
SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手。通过 run_bash 工具在沙箱里执行 fincli 命令查 A 股年报与天气。"
    "查年报：fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3"
    "（query 不要含公司名/年份，用简短财务术语）。"
    "列公司：fincli list-companies。"
    "查天气：fincli weather --city 南京。"
    "回答必须依据命令返回的原文，不要编造。知识库仅含 5 家公司（茅台/五粮液/宁德时代/海康威视/中国平安），"
    "不在库内的明确告知。本回合可一次调用多个工具。"
)
```

**讲解**：形态 B 的 System Prompt 额外提供了 fincli 命令的具体语法示例，因为 LLM 需要自己拼 shell 命令。这是**Prompt Engineering 中的 Few-Shot 教学**——通过示例教会 LLM 如何正确使用工具。

### 2.10 单轮闭环 `run()` 函数

```python
def run(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    tools_schema, executor = MODE_DISPATCH[mode]  # 根据 mode 选择 schema 和执行函数
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []  # 记录所有 tool call，用于日志和调试
```

**讲解**：构建标准的 messages 列表——System Prompt 设定角色和规则，user message 提出问题。

```python
    # ── 第一次 LLM 调用：让 LLM 决策要不要调工具 ──
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools_schema, tool_choice="auto",
    )
    msg = resp.choices[0].message
```

**讲解**：
- `tools=tools_schema` 把工具列表传给 LLM
- `tool_choice="auto"` 表示让 LLM 自己决定是否要调用工具（也可以设为 `"required"` 强制调用或 `"none"` 禁用）
- 返回的 `msg` 可能包含 `tool_calls`，也可能包含直接回答的文本 `content`

```python
    if msg.tool_calls:  # LLM 决定调用工具
        messages.append(msg)  # 把 assistant 的 tool_calls 加到消息历史
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")  # 解析 JSON 参数
            tool_call_log.append({"name": tc.function.name, "args": args})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")

            try:
                result = executor(args)  # 在 host 上执行工具
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
```

**讲解**：这是标准的 OpenAI Function Call 循环：

1. 把 LLM 的 tool_calls 消息添加到 messages 历史（保持对话连续性）
2. 遍历每个 tool_call：
   - 解析 arguments JSON
   - 调用 executor（host 端执行）
   - 把执行结果作为 `role: "tool"` 消息追加到 messages

> **注意**：`executor(args)` 中的 `args` 是直接来自 LLM 的 JSON 对象。在形态 A 中，executor 是 `lambda args: run_named(args["command"], args.get("args", {}))`，会从 args 中提取 command 和子参数。

```python
        # ── 第二次 LLM 调用：让 LLM 基于工具结果生成最终回答 ──
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message
```

**讲解**：第二轮调用让 LLM 消化工具执行的结果，生成面向用户的自然语言回答。"单轮闭环"指的就是：用户提问 → LLM 调工具 → 获取结果 → LLM 回答。如果需要更多轮工具调用（比如先 list-companies 再 search），`tool_choice="auto"` 会自动触发。

```python
    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}
```

**讲解**：返回结构化的结果字典，包含三个字段：

| 字段 | 含义 | 用途 |
|------|------|------|
| `answer` | LLM 的最终回答文本 | 展示给用户 |
| `tool_calls` | 工具调用记录 | 调试、审计 |
| `elapsed` | 总耗时 | 性能分析 |

### 2.11 入口部分

```python
DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",                                            # 单公司单年份
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",                         # 跨工具调用
    "对比贵州茅台和五粮液2023年的营收。",                                             # 多公司比较
    "比亚迪2023年营收是多少？",                                                       # 不在库内的公司
]
```

**讲解**：四个 Demo 问题精心设计，分别测试不同能力：

| # | 测试能力 | 预期 LLM 行为 |
|---|----------|---------------|
| 1 | 基础检索 | 调一次 `rag_search` |
| 2 | 跨工具 | 先调 `rag_search` 再调 `weather` |
| 3 | 多轮调用 | 调两次 `rag_search`（茅台 + 五粮液） |
| 4 | 边界处理 | 先调 `list_companies` 确认比亚迪不在库内，然后告知用户 |

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式三：CLI")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()
```

**讲解**：命令行参数：

| 参数 | 简写 | 默认 | 含义 |
|------|------|------|------|
| `--mode` | — | `named` | 形态 A/B 切换 |
| `--question` | `-q` | — | 用户问题 |
| `--demo` | — | `False` | 运行内置示例 |
| `--provider` | — | `deepseek` | LLM Provider |
| `--quiet` | — | `False` | 安静模式（减少输出） |
| `--json` | — | `False` | JSON 输出（供 `compare.py` 脚本解析） |

```python
    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n", file=sys.stderr)
```

**讲解**：初始化 LLM 客户端。注意日志打印到 `sys.stderr` 而非 stdout——因为 `--json` 输出时 stdout 要保持干净（只输出合法的 JSON）。

```python
    # 问题优先级：--demo > --question > 默认第一个 demo 问题
    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])

    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)

        result = run(client, model, q, args.mode, verbose=not (args.quiet or args.json))
        result["question"] = q
        result["mode"] = args.mode
        results.append(result)

        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
```

**讲解**：
- 问题选择逻辑用 Python 的惯用写法：`A if condition else (B if condition2 else C)`
- `enumerate(questions, 1)` 从 1 开始编号
- `json.dumps(..., ensure_ascii=False)` 保证中文正常显示
- 单个问题时输出对象，多个问题时输出数组

```python
if __name__ == "__main__":
    main()
```

---

## 三、整体架构回顾

将两个文件放在一起看，可以清楚地看到分层设计：

```
┌─────────────────────────────────────────────────────────────┐
│  run_cli.py  ── LLM 意图生成层                               │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ 形态 A (named)   │  │ 形态 B (bash)   │                   │
│  │ LLM → run_cli() │  │ LLM → run_bash()│                   │
│  │ host 拼命令      │  │ LLM 拼命令       │                   │
│  │ 最安全           │  │ 沙箱保护         │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│           └────────┬───────────┘                             │
│                    ▼                                         │
│           subprocess.run()                                   │
├─────────────────────────────────────────────────────────────┤
│  main.py  ── 工具实现层 (fincli)                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  argparse 子命令：search / list-companies / weather    │  │
│  │  → src.rag_backend / src.weather_backend               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 三种方式对比

| 维度 | 方式一 Function Call | 方式二 MCP | 方式三 CLI |
|------|---------------------|-----------|-----------|
| 工具调用方式 | 直接调用 Python 函数 | MCP JSON-RPC 协议 | subprocess 执行命令行 |
| LLM 交互 | OpenAI tools API | MCP list_tools + call_tool | Function Call 包装 CLI |
| 安全性 | 高（函数调用） | 高（协议定义） | 取决于形态 |
| 扩展性 | 中 | 高（支持多语言） | 高（任何语言） |
| 适用场景 | Python 项目 | 跨语言/跨进程 | 现有 CLI 工具 |

**核心关系**：这三种方式不是互斥的，而是**不同层次**的技术：
- `src/` 是 n 个普通 Python 函数（后端实现）
- `main.py` CLI 是对这些函数的一个 argparse 封装（工具实现层）
- `run_cli.py` 是让 LLM 通过 Function Call 调用 `main.py` CLI（意图生成层）
- `rag_server.py` 是让 LLM 通过 MCP 协议调用 `src/` 后端函数（协议接入层）

三者共享同一个 `src/` 后端，实现了"**一次实现，多种接入**"的架构设计。
