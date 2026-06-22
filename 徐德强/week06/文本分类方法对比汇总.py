实验总结 — 文本分类方法对比
============================
日期：2026-05-29
任务：TNEWS 15 分类

一、结果汇总

┌──────────────────┬──────────┬──────────┬──────────┬────────────┐
│ 方法             │ Accuracy │ Macro F1 │ 训练用时 │ 训练数据   │
├──────────────────┼──────────┼──────────┼──────────┼────────────┤
│ BERT-CLS         │  0.5657  │  0.5528  │  2452s   │ 53K 条     │
│ BERT-MAX         │  0.5641  │  0.5558  │  1660s   │ 53K 条     │
│ BERT-MEAN        │  0.5648  │  0.5532  │  1593s   │ 53K 条     │
│ Zero-shot        │  0.3600  │   —      │  0s      │ 0 条       │
│ Few-shot (5-shot)│  0.3250  │   —      │  0s      │ 75 条示例  │
│ LoRA 微调        │  0.4850  │   —      │  485s     │ 2000 条    │
└──────────────────┴──────────┴──────────┴──────────┴────────────┘

二、关键发现

1. BERT 组显著优于 LLM 组
   - 三种 BERT 池化方式准确率均在 56.5% 左右，远超所有 LLM 方法。
   - BERT 有 53K 条标注数据，LLM 组最多只用了 2000 条（LoRA）。

2. BERT 三种池化策略差异极小
   - CLS / Max / Mean 准确率差距在 0.002 以内，几乎无差别。
   - Mean 和 Max 池化的训练速度比 CLS 快约 35%（1593s vs 2452s）。

3. Few-shot 低于 Zero-shot（异常）
   - Zero-shot: 36.0%，Few-shot (5-shot): 32.5%。
   - 仅 5 个示例可能不够，甚至引入偏差。建议尝试 10-shot 或 20-shot。

4. LoRA 微调提升明显但仍有差距
   - LoRA 相比 Zero-shot 提升 12.5 个百分点（36% → 48.5%）。
   - 但离 BERT 仍差约 8 个点，说明 2000 条训练数据还不够。

三、结论与建议

- 有足够标注数据时，BERT 微调是首选，训练快且效果好。
- LLM 在零标注场景可用（36% 作为 baseline），但不要对 Few-shot 抱过高期望。
- 要追上 BERT，LoRA 需要更多训练数据或更大参数量模型。
- 下一步可以尝试：增加 n_shot 数量、扩大 LoRA 训练集、换更大的 LLM（如 Qwen2.5-1.5B）。

四、文件清单

- BERT 权重：outputs/checkpoints/best_{cls,max,mean}.pt
- 训练日志：outputs/train_log_{cls,max,mean,sft}.json
- LLM 结果：outputs/llm_{zero_shot,fewshot,sft}_results.json
- 对比图表：outputs/figures/methods_comparison.png
