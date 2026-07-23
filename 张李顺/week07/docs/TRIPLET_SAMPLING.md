# Triplet Loss 负样本取样策略

每个三元组是 `(anchor, positive, negative)`：

- `anchor`：正标签行的 `sentence1`。
- `positive`：同一行的 `sentence2`。
- `negative`：同一个 `anchor` 在原始训练集中明确标为 0 的 `sentence2`。

只使用同 anchor 的显式负例，不再从全局语料随机补负例，因此不会把“未标注关系”擅自当作负例。

如果一个 anchor 有多个显式负例，则计算字符 unigram+bigram Jaccard 重合度，选择与 anchor 词面最接近的负例作为 hard negative。这里遍历该 anchor 的全部显式负例，没有固定候选数 16。

没有显式负例的正样本不进入 Triplet 训练。LCQMC 有 19,422 条正样本具备同 anchor 显式负例，BQ 有 27,211 条，均足以覆盖当前 8,192 条训练预算。

训练使用余弦距离 `1-cosine(a,b)` 和 margin 0.2 的 `TripletMarginWithDistanceLoss`。随机种子固定为 42，只决定从合格三元组中抽取哪些样本。
