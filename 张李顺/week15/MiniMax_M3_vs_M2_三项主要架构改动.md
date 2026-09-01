# MiniMax M3 相对 M2 的三项主要架构改动

## 1. Attention：Full GQA → MSA

### 组件位置

位于 Transformer Block 的 **Attention 子层**。

M2 使用的是 **Full GQA（Grouped Query Attention）**。  
M3 保留 GQA 的基本结构，在其上增加 **MSA（MiniMax Sparse Attention）**，核心新增模块是 **Indexer**。

---

### M2 的具体问题

M2 的 Full GQA 会让当前 Query 对完整历史上下文中的 K/V 进行 Attention。

随着上下文长度增加，主要出现三个问题：

1. **Prefill 阶段 Attention 计算量快速增长**

   对长度为 `N` 的输入序列，完整 Attention 需要计算大量 Query-Key 配对。

   序列越长，需要计算的相关性矩阵越大。百万 Token 场景下，Attention FLOPs 会成为非常大的计算负担。

2. **Decode 阶段需要读取越来越长的 KV Cache**

   自回归生成时，每生成一个新 Token，都需要使用当前 Query 与历史 K/V 做 Attention。

   历史上下文越长，需要从显存中读取的 K/V 数据越多，因此 KV Cache 的显存带宽压力持续增加。

3. **长上下文推理延迟增加**

   Full Attention 会让大量与当前 Query 关系较弱的历史 Token 仍然参与计算。

   当上下文达到几十万甚至上百万 Token 时，这部分计算和 KV 读取会直接转化为更高的 Prefill 延迟和 Decode 延迟。

---

### M3 的改动

M3 在 GQA 上增加了 **MSA（MiniMax Sparse Attention）**。

整体逻辑可以简化为：

```text
当前 Query
   ↓
Indexer
   ↓
从完整历史 KV 中寻找相关区域
   ↓
选出少量 KV Blocks
   ↓
只对这些 KV Blocks 做真正的 Softmax Attention
```

M3 仍然保留 GQA：

```text
多个 Q Head
   ↓
共享一个 KV Head
```

GQA 解决的是 **KV Head 数量过多带来的 KV Cache 开销**。

MSA 继续在这个基础上缩小 Attention 实际读取的历史范围。

---

### KV Block 是什么

KV Block 是按照 **Token 位置** 对历史 K/V 进行分块。

例如每 128 个 Token 为一个 Block：

```text
Block 1：Token 1 ~ 128 对应的 K/V
Block 2：Token 129 ~ 256 对应的 K/V
Block 3：Token 257 ~ 384 对应的 K/V
...
```

它和 GQA 的“组”属于两个不同维度：

```text
GQA Group：按 Attention Head 分组
KV Block： 按历史 Token 位置分组
```

一个 GQA Group 对应的历史 K/V 中，可以包含大量 KV Blocks。

---

### Indexer 的作用

Indexer 是 Attention 内部增加的一个轻量可训练检索分支。

它不负责生成 Token，也不单独完成完整模型推理。

它的任务只有一个：

> 判断当前 Query 最值得读取哪些历史 KV Blocks。

逻辑上是 **一个 Indexer 模块**，内部针对不同 GQA Group 使用不同的 Index Query，因此不同 GQA Group 可以得到不同的 KV Block 选择结果。

同时，Index Key 表示可以共享。

简化结构如下：

```text
                 ┌─ GQA Group 1 → 自己的 Index Query ─┐
Hidden State ────┼─ GQA Group 2 → 自己的 Index Query ─┤
                 ├─ GQA Group 3 → 自己的 Index Query ─┤
                 └─ GQA Group 4 → 自己的 Index Query ─┘
                               ↓
                         共享 Index Key
                               ↓
                    每组得到自己的 KV Block
                               ↓
                    真正的 Softmax Attention
```

---

### 解决思路

M2：

```text
当前 Query
   ↓
读取全部历史 KV
   ↓
对全部历史 Token 做 Attention
```

M3：

```text
当前 Query
   ↓
Indexer 先做轻量检索
   ↓
从完整历史 KV 中筛选少量相关 Block
   ↓
只读取这些 Block 的 K/V
   ↓
继续使用精确 Softmax Attention
```

核心思想是：

> **保留 GQA 和标准 Softmax Attention 的计算方式，同时减少真正参与 Attention 的历史 K/V 范围。**

因此 M3 的 Attention 优化重点并不是简单更换一种新的近似 Attention，而是先增加一个检索步骤，让主 Attention 只处理相关历史区域。

---

## 2. FFN / MoE：256 小 Expert Top-8 → 128 大 Expert Top-4 + Shared Expert

### 组件位置

位于 Transformer Block 的 **FFN / MoE 子层**。

M2 和 M3 都使用 MoE，但 Expert 组织方式发生了较大的变化。

---

### M2 的结构

M2：

```text
256 个 Routed Experts
Top-8 Routing
```

每个 Token 到达 MoE 层后，Router 会从 256 个 Expert 中选择 8 个。

因此：

```text
1 个 Token
   ↓
Router
   ↓
8 个 Expert
```

一个 Token 会产生 **8 个 Token-Expert Assignment**。

---

### M2 的具体问题

#### 1. 每个 Token 的 Routed Expert 数量较多

Top-8 意味着一个 Token 同时需要被发送给 8 个 Expert。

如果某一层输入有 `T` 个 Token，则理论上会形成约：

```text
T × 8
```

个 Routed Token-Expert Assignment。

Assignment 数越多，MoE 系统需要处理的路由、打包、发送和聚合操作越多。

---

#### 2. Expert 分布在多 GPU 时，通信 Fan-out 较大

大型 MoE 模型通常不会把全部 Expert 放在同一张 GPU 上。

Expert 会分散在多张 GPU 或多个设备上。

例如一个 Token 被 Router 选择了 8 个 Expert：

```text
Token
 ├→ Expert A
 ├→ Expert B
 ├→ Expert C
 ├→ Expert D
 ├→ Expert E
 ├→ Expert F
 ├→ Expert G
 └→ Expert H
```

如果这些 Expert 位于不同 GPU，就需要把这个 Token 的 Hidden State 分发到对应设备。

因此 Top-8 会带来较大的 **跨设备通信 Fan-out**。

这里需要区分两个概念：

- Expert 总数量：256
- 每个 Token 实际访问 Expert 的数量：8

真正直接决定单 Token 路由 Fan-out 的主要是 **Top-K**。

---

### M3 的改动

M3 改成：

```text
128 个 Routed Experts
Top-4 Routing
+
1 个 Shared Expert
```

同时，单个 Routed Expert 的规模比 M2 更大。

可以理解为：

```text
M2
256 个较小 Expert
每 Token → Top-8

        ↓

M3
128 个更大的 Routed Expert
每 Token → Top-4
+
所有 Token → Shared Expert
```

---

### Top-4 带来的变化

M2：

```text
1 Token → 8 Routed Experts
```

M3：

```text
1 Token → 4 Routed Experts
```

因此 Routed 部分的 Token-Expert Assignment 数量直接减半。

例如 1000 个 Token：

```text
M2：
1000 × 8 = 8000 个 Routed Assignment

M3：
1000 × 4 = 4000 个 Routed Assignment
```

在 Expert Parallel 场景下，这通常意味着需要跨设备分发的 Routed Token 副本更少，通信 Fan-out 更低。

---

### 为什么同时增加 Shared Expert

如果只是把 Top-8 降成 Top-4，每个 Token 能访问的 Routed Expert 数量也会减少。

M3 增加一个 **Shared Expert**：

```text
                 ┌→ Shared Expert ─────────┐
Token ───────────┤                         ├→ 合并输出
                 └→ Top-4 Routed Experts ─┘
```

Shared Expert 对所有 Token 都执行。

它可以承担模型中大量通用、重复出现的特征处理。

Routed Experts 则继续承担由 Router 决定的差异化计算。

因此 M3 的 MoE 组织方式可以概括为：

```text
通用能力
   ↓
Shared Expert

差异化能力
   ↓
Top-4 Routed Experts
```

---

### 解决思路

M2：

```text
256 个小 Expert
Top-8
   ↓
每个 Token 产生 8 个 Routed Assignment
   ↓
通信 Fan-out 较大
```

M3：

```text
128 个更大的 Routed Expert
Top-4
+
1 Shared Expert
   ↓
Routed Assignment 减少
   ↓
通用信息由 Shared Expert 承担
```

核心思想是：

> **降低每个 Token 的 Routed Expert 数量，用更大的 Routed Expert 和 Shared Expert 重新分配计算职责。**

这里不能简单理解为“Top-8 改 Top-4，所以总计算量减半”。

M3 的单个 Expert 更大，同时增加了 Shared Expert，因此总 FLOPs 需要结合完整 Expert 维度计算。

能够直接确定的变化是：

> **Routed Token-Expert Assignment 从每 Token 8 个降到 4 个，Routed 通信 Fan-out 随之下降。**

---

## 3. 多模态输入：文本模型 → 原生 Text / Image / Video 输入

### 组件位置

位于 Transformer 主干之前的 **输入编码组件**。

M2 的主模型以文本 Token 作为输入。

M3 增加完整视觉输入路径，使图像和视频能够转换为与文本模型兼容的视觉 Token，再进入统一语言模型主干。

---

### M2 的具体问题

M2 的基础输入空间主要面向文本：

```text
Text
 ↓
Tokenizer
 ↓
Text Token
 ↓
Embedding
 ↓
Transformer
```

因此模型主干天然处理的是离散文本 Token。

图像和视频本身不能直接作为文本 Embedding 输入。

如果需要处理视觉信息，需要额外的视觉编码流程将图像转换为模型可以使用的表示。

---

### M3 的改动

M3 增加了完整的视觉输入链路：

```text
Image / Video
      ↓
Vision Encoder
      ↓
Visual Tokens
      ↓
Projector
      ↓
映射到 LLM Hidden Size
      ↓
与 Text Tokens 一起进入 Transformer
```

因此整个输入结构变为：

```text
Text ─────→ Text Embedding ───────┐
                                  │
Image ─┐                          │
       ├→ Vision Encoder          ├→ Transformer 主干
Video ─┘       ↓                  │
           Visual Tokens          │
                ↓                 │
             Projector ───────────┘
```

---

### Vision Encoder 的作用

Vision Encoder 负责把原始图像或视频转换为视觉特征。

原始图片是二维像素矩阵，无法直接送进语言模型。

Vision Encoder 会先把图像切成 Patch，再把这些 Patch 转换为一系列 Visual Tokens：

```text
Image
 ↓
Patch 切分
 ↓
Patch Embedding
 ↓
Vision Transformer
 ↓
Visual Tokens
```

视频可以进一步保留时间维度的信息。

---

### Projector 的作用

视觉编码器输出的特征维度和语言模型 Hidden Size 不一定一致。

Projector 负责把视觉特征映射到语言模型可以接受的表示空间：

```text
Vision Hidden State
        ↓
     Projector
        ↓
LLM Hidden Dimension
```

完成投影后，视觉 Token 就可以和文本 Token 一起进入后续 Transformer Block。

---

### 解决思路

M2：

```text
Text Token
   ↓
Transformer
```

M3：

```text
Text ───────────────┐
                    │
Image → Vision ─────┤
                    ├→ 统一 Transformer
Video → Vision ─────┘
```

核心变化是：

> **在 Transformer 主干前增加视觉编码和模态对齐组件，把 Image / Video 转换成统一的 Token 表示。**

因此 M3 的基础模型输入不再局限于文本 Token，可以在同一个模型主干中处理文本、图片和视频信息。

---

# 总结

MiniMax M3 相比 M2，主要有三项值得关注的大规模架构变化。

| Transformer 位置 | M2 的具体问题 | M3 的主要改动 | 直接解决方向 |
|---|---|---|---|
| **Attention** | Full GQA 在长上下文中需要对大量历史 K/V 做 Attention；Prefill FLOPs、Decode KV 读取量和长上下文延迟持续增加 | **GQA + MSA + Indexer + KV Block 筛选** | 缩小真正参与 Attention 的历史 KV 范围 |
| **FFN / MoE** | Top-8 使每个 Token 产生 8 个 Routed Token-Expert Assignment，Expert Parallel 下通信 Fan-out 较大 | **256 小 Expert / Top-8 → 128 大 Expert / Top-4 + Shared Expert** | 减少 Routed Assignment 和跨设备通信 Fan-out |
| **输入编码** | M2 主模型主要处理文本 Token，图像和视频缺少统一的原生输入路径 | **Vision Encoder + Projector + Text/Image/Video 统一输入** | 把视觉信息转换为与语言模型兼容的 Visual Tokens |

如果只从架构研究价值排序：

```text
1. MSA / Indexer / KV Block Sparse Attention
2. MoE 粒度与 Shared Expert 重构
3. 原生多模态输入链路
```

其中第一项是 M3 相对 M2 最核心、最明确的 Transformer 内部架构变化。
