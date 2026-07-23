"""
main.py — fincli：AI技术面试知识检索 + 天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli list-papers
  fincli search --query "自注意力机制" --topic "Transformer架构" --top-k 3
  fincli weather --city 北京

不想安装也可直接跑：
  python mode_cli/cli/main.py search --query "自注意力机制" --topic "Transformer架构" --top-k 3
  python -m mode_cli.cli.main weather --city 北京

核心设计：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本——跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含多个子命令（search/list-papers/weather），对应 git 的子命令设计

依赖：
  pip install faiss-cpu numpy openai httpx
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""

import argparse
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_backend import search_ai_knowledge, list_papers  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — AI技术面试知识检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli search ...
    p_search = sub.add_parser("search", help="检索AI技术知识段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含论文标题/主题，用简短技术术语，如 '自注意力机制'）")
    p_search.add_argument("--title", default=None, help="按论文标题过滤，如 'Attention Is All You Need'")
    p_search.add_argument("--topic", default=None, help="按主题过滤：Transformer架构/预训练语言模型/大语言模型/指令微调/开源大模型/检索增强生成/深度学习教程")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")

    # fincli list-papers
    sub.add_parser("list-papers", help="列出知识库收录的论文")

    # fincli weather ...
    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 北京")

    args = parser.parse_args()

    if args.cmd == "search":
        print(search_ai_knowledge(args.query, args.title, args.topic, args.top_k))
    elif args.cmd == "list-papers":
        print(list_papers())
    elif args.cmd == "weather":
        print(get_weather(args.city))


if __name__ == "__main__":
    main()