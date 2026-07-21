# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/7/16 19:20  
@脚本说明 : 

"""
#!/usr/bin/env python3
"""
llm_chat.py — 命令行大模型问答系统
支持 OpenAI API 及兼容服务（Ollama、vLLM、DeepSeek 等）
环境变量：
    OPENAI_API_KEY   (必需，若使用 OpenAI)
    OPENAI_BASE_URL  (可选，默认 https://api.openai.com/v1)
    DEFAULT_MODEL    (可选，默认 gpt-3.5-turbo)
交互命令：
    /clear           清空对话历史
    /model <name>    切换模型
    /exit            退出程序
"""

import os
import sys
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# 初始化 rich 控制台（使输出更美观）
console = Console()

class ChatSession:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            console.print("[red]错误：未设置 OPENAI_API_KEY，请设置环境变量或在命令行输入[/red]")
            sys.exit(1)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.history = []  # 存储对话消息 [{"role": "user"/"assistant", "content": ...}]

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history = []
        console.print("[yellow]对话历史已清空[/yellow]")

    def set_model(self, model_name):
        self.model = model_name
        console.print(f"[green]切换模型至：{model_name}[/green]")

    def chat(self, user_input):
        """发送用户输入，获取流式回复并显示"""
        self.add_message("user", user_input)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True,
            )
            console.print("\n[bold cyan]助手：[/bold cyan]", end="")
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    console.print(content, end="")
                    full_response += content
            console.print("\n")  # 换行
            self.add_message("assistant", full_response)
        except Exception as e:
            console.print(f"[red]请求失败：{e}[/red]")
            # 不将错误信息加入历史

def main():
    console.print("[bold blue]🚀 命令行大模型问答系统[/bold blue]")
    console.print(f"当前模型：[cyan]{os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')}[/cyan]")
    console.print("输入问题开始对话，输入 [yellow]/clear[/yellow] 清空历史，[yellow]/model <name>[/yellow] 切换模型，[yellow]/exit[/yellow] 退出\n")

    # 允许用户在启动时输入 API Key（如果未设置环境变量）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("请输入您的 OpenAI API Key（或按回车使用环境变量，若无请设置）：").strip()
        if not api_key:
            console.print("[red]未提供 API Key，程序退出。[/red]")
            return

    session = ChatSession(api_key=api_key)

    while True:
        try:
            user_input = input("[bold green]你：[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold red]退出。[/bold red]")
            break

        if not user_input:
            continue

        # 处理内置命令
        if user_input.lower() == "/exit":
            console.print("[bold red]再见！[/bold red]")
            break
        elif user_input.lower() == "/clear":
            session.clear_history()
            continue
        elif user_input.startswith("/model "):
            model_name = user_input[7:].strip()
            if model_name:
                session.set_model(model_name)
            else:
                console.print("[yellow]请指定模型名称，例如：/model gpt-4[/yellow]")
            continue
        elif user_input.startswith("/"):
            console.print(f"[yellow]未知命令：{user_input}，可用命令：/clear, /model, /exit[/yellow]")
            continue

        # 正常问答
        session.chat(user_input)

if __name__ == "__main__":
    main()