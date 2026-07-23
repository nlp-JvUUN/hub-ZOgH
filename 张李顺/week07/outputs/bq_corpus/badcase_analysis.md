# bq_corpus Badcase

测试样本：8618。CSV 保存全部预测错误，不做自动原因分类。

## bi_cosine

- FP：1536
- FN：511
- 总错误：2047
- 错误率：0.2375

## bi_triplet

- FP：1272
- FN：563
- 总错误：1835
- 错误率：0.2129

## CrossEncoder

- FP：1292
- FN：560
- 总错误：1852
- 错误率：0.2149

## BiEncoder + BM25

- FP：1173
- FN：699
- 总错误：1872
- 错误率：0.2172

## LLM Zero-shot

- FP：2494
- FN：672
- 总错误：3166
- 错误率：0.3674

## LLM SFT LoRA

- FP：1526
- FN：767
- 总错误：2293
- 错误率：0.2661
