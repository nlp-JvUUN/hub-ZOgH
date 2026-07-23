"""
main.py — fincli：A股年报检索 + 天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli list-companies
  fincli search --query "营收和净利润" --stock-code 300750 --year 2023 --top-k 3
  fincli weather --city 宁德

不想安装也可直接跑：
  python mode_cli/cli/main.py search --query "营收" --stock-code 300750 --year 2023
  python -m mode_cli.cli.main weather --city 宁德

教学点：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本——跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含多个子命令（search/list-companies/weather），对应 git 的子命令设计

依赖：
  pip install faiss-cpu numpy openai httpx
  环境变量：DASHSCOPE_API_KEY（Embedding）

流程分析：
用户问题
    ↓
LLM（看过 system prompt，知道 fincli 怎么用）
    ↓ 判断：需要查年报 → 生成命令 "fincli search --query '营收' --stock-code 300750 --year 2023"
    ↓ 判断：需要查天气 → 生成命令 "fincli weather --city 宁德"
    ↓
Host 执行 subprocess.run("fincli search ...")
    ↓
fincli（cli/main.py）解析命令行参数
    ↓ 调用 src.rag_backend.search_annual_report() 或 src.weather_backend.get_weather()
    ↓
结果通过 stdout 返回
    ↓
Host 把结果回填给 LLM → 生成最终回答

stdin 和 stdout 是标准输入/输出流，操作系统给每个进程默认打开的三条"通道"之一：

名称	   含义	    用途
stdin	  标准输入	进程从键盘或管道读取数据
stdout	标准输出	进程向屏幕或管道输出正常内容
stderr	标准错误	进程向屏幕输出错误信息

在 MCP 协议里，stdio_client 就是通过这两个通道和 Server 子进程通信：
Host (stdin)  ──────→  Server
Host (stdout) ←──────  Server

因为 MCP Server 是子进程，不能像本地函数那样直接调——所以通过操作系统提供的 
stdin/stdout 管道传递 JSON 格式的请求和响应，模拟"函数调用"的效果。这是一种**进程间通信（IPC）**方式。

先写好 pyproject.toml，定义好 fincli = "mode_cli.cli.main:main" 这个入口点
再执行 pip install -e .，setuptools 读取配置，在 PATH 上生成 fincli 命令
之后就能在任意目录直接敲 fincli search ... 调用了
pip install -e .（editable 模式）的好处是：改 cli/main.py 代码后不需要重装， fincli 命令始终指向最新版本。

fincli.egg-info是 pip install -e . 时 setuptools 自动生成的安装元数据目录。
这个目录记录了安装信息，让 pip 知道包的结构和入口点。主要文件：
文件	              内容
entry_points.txt	入口点配置，fincli = mode_cli.cli.main:main
PKG-INFO	        包名/版本/描述等元信息
SOURCES.txt	      装进去了哪些源文件
requires.txt	    依赖包列表
和 cli/main.py 的执行无关，只是 pip install -e . 的副产品。如果删掉它，重新 pip install -e . 就会重新生成。
"""

import argparse
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli search ...
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含公司名/年份，用简短财务术语，如 '营收和净利润'）")
    p_search.add_argument("--stock-code", default=None, help="按公司过滤，如 300750")
    p_search.add_argument("--year", default=None, help="按年份过滤：2021/2022/2023")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")

    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录的公司")

    # fincli weather ...
    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")

    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "weather":
        print(get_weather(args.city))


if __name__ == "__main__":
    main()
