# 三方式对比结果（Function Call / MCP / CLI）

- LLM provider：`deepseek`
- 生成时间：本表由 `python compare.py` 实跑生成
- 问题数：4，方式数：4

## 对比表

| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 正确拒绝幻觉 | 答案摘要 |
|------|------|---------|:------:|:-------:|:------------:|---------|
| Transformer的自注意力机制原理是什么？ | Function Call | search_ai_knowledge, search_ai_knowledge | 2 | 14.0s | - | 以下是基于论文原文对 **Transformer 自注意力机制**的详细原理介绍：  ---  ## 自注意力机制（Self-Attention）原理详解  #... |
| Transformer的自注意力机制原理是什么？ | MCP | search_ai_knowledge | 1 | 14.2s | - | 根据《Attention Is All You Need》（2017）论文原文，我来详细解释Transformer的自注意力机制（Scaled Dot-Prod... |
| Transformer的自注意力机制原理是什么？ | CLI(named) | run_cli | 1 | 17.4s | - | 根据检索到的知识库内容，我来为你详细讲解Transformer的自注意力机制原理。  ---  ## Transformer自注意力机制原理  ### 1. 什... |
| Transformer的自注意力机制原理是什么？ | CLI(bash) | run_bash, run_bash, run_bash, run_bash | 4 | 40.4s | - | 好的，现在所有信息都已检索完毕。下面给出完整的回答。  ---  ## 一、Transformer的自注意力机制原理  根据知识库中《动手学深度学习》的讲解，自... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | Function Call | search_ai_knowledge, get_weather | 2 | 14.6s | - | ## Transformer 自注意力机制原理  根据《Attention Is All You Need》论文中的内容，自注意力（Self-Attention... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | MCP | search_ai_knowledge, get_weather | 2 | 14.6s | - | ## 1️⃣ Transformer的自注意力（Self-Attention）原理  根据《Attention Is All You Need》（2017）原文... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | CLI(named) | run_cli, run_cli | 2 | 23.7s | - | 以下是两个问题的回答：  ---  ## 一、Transformer的自注意力机制原理  自注意力（Self-Attention），也称为 **intra-at... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | CLI(bash) | run_bash, run_bash, run_bash, run_bash | 4 | 37.7s | - | 好的，现在我已经获取了足够的信息，下面给出完整回答。  ---  ## 一、Transformer自注意力机制的原理  根据检索到的资料，自注意力（self-a... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | Function Call | search_ai_knowledge, search_ai_knowledge, list_papers, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge | 7 | 18.7s | - | 好的，现在我已经从知识库中检索到了足够的信息，可以为您进行全面的对比分析。  ---  ## BERT 与 GPT-3 预训练方式的对比  基于知识库中论文原文... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | MCP | search_ai_knowledge, search_ai_knowledge, list_papers, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge | 9 | 26.6s | - | 好的，我已经从知识库中检索到了相关段落，下面基于检索到的内容，为您详细对比BERT和GPT-3的预训练方式。  ---  ## BERT vs GPT-3：预训... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | CLI(named) | run_cli, run_cli, run_cli | 3 | 30.7s | - | 根据检索到的资料，下面系统地对比BERT和GPT-3（以及其前身OpenAI GPT）在预训练方式上的主要区别：  ---  ## 一、核心架构差异：双向 vs... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | CLI(bash) | run_bash, run_bash, run_bash | 3 | 31.4s | - | 好的，现在让我基于知识库中已有的 **BERT** 和 **GPT-3** 论文内容，为你详细对比两者的预训练方式差异。  > 注：知识库中 "自注意力机制" ... |
| 区块链的共识算法是什么？ | Function Call | (无工具调用) | 0 | 3.3s | ✓ 拒绝 | 我无法回答这个问题，因为您询问的"区块链的共识算法"不属于我的知识库收录范围。我的知识库仅涵盖以下主题：  1. **Transformer架构**（Atten... |
| 区块链的共识算法是什么？ | MCP | (无工具调用) | 0 | 5.7s | ✗ 未拒绝(可能幻觉) | 您咨询的"区块链的共识算法"不属于我的知识库范围。我目前的知识库仅包含以下主题：  - Transformer架构（Attention Is All You N... |
| 区块链的共识算法是什么？ | CLI(named) | (无工具调用) | 0 | 3.9s | ✓ 拒绝 | 抱歉，关于“区块链的共识算法”这一主题，不在我的知识库覆盖范围内。  我的知识库仅包含以下主题： - Transformer架构 - 预训练语言模型 - 大语言... |
| 区块链的共识算法是什么？ | CLI(bash) | run_bash | 1 | 10.5s | ✓ 拒绝 | 知识库中收录的 **7 份文档**全部是 AI / 深度学习 / 大模型相关的内容，分别是：  1. **Attention Is All You Need**... |

## 解读

- **工具调用一致性**：四种方式对同一问题调用的工具与参数基本一致——说明底层能力相同，差异在『接入方式』而非『能力』。
- **接入成本**：Function Call 要手写 schema；MCP 要写 Server 但工具自动发现可跨产品复用；CLI(named) 写白名单；CLI(bash) 几乎零封装但需沙箱。
- **安全**：Function Call / MCP / CLI(named) 都走白名单，安全；CLI(bash) 依赖沙箱拦截，最危险。
- **跨模型复用**：MCP 工具可被任意支持 MCP 的 Host 复用；Function Call schema 各家 API 略有差异；CLI 与模型完全无关。
- **幻觉控制**：问区块链（不在知识库）时，看各方式是否正确拒绝而非编造数据。

## 各方式原始回答
