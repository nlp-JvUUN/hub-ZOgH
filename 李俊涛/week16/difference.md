MiniMax

注意力：Lightning 线性注意力（右积核技巧，复杂度 O(L)），每 8 层中 7 层线性 + 1 层 Softmax 全量注意力
FFN：MoE，32 路由专家 / 每 token top-2
归一化：RMSNorm（线性层内另有 block 级归一化、分块、衰减）
位置编码：RoPE
规模/上下文：456B 总参 / 45.9B 激活，原生 1M、外推 4M
多模态：VL-01 接 ~303M ViT + 两层 projector
腾讯混元 Hunyuan

Hunyuan-Large：MoE = 1 共享专家 + 16 专用专家，每 token 激活 1 专用 + 1 共享（top-1）；GQA + 跨层注意力 CLA 压缩 KV（降约 95%）；专家专属学习率（专用 = 共享×0.31）；RoPE（256K 阶段基频扩至 1e9）；389B/52B，256K
TurboS / T1：混合 Transformer-Mamba，128 层 = 57 Mamba2 + 7 Attention + 64 FFN-MoE(32 专家)，AMF/MF 块交替；560B/56B，256K
Meta Llama 4（Scout / Maverick）

注意力：iRoPE——NoPE 全量层（每 4 层，无位置编码、全因果掩码）+ 分块 RoPE 层（3/4 层，窗口 8192）；NoPE 层推理时做注意力温度缩放 → Scout 推到 10M
FFN：MoE，Scout 16 专家 / 17B 激活 / 109B；Maverick 128 专家 / 17B 激活 / 400B
归一化：RMSNorm + QK-Norm（RoPE 层，无学习参数）
训练：FP8；MetaP 跨规模超参迁移；Behemoth 共蒸馏
多模态：early fusion，文本与图像同 backbone
Google Gemma 3

注意力：5:1 局部滑窗 + 全局交替（5 层窗口 1024 + 1 层全量，首层即局部）
FFN：dense GeGLU（无 MoE）
归一化：每层 4× RMSNorm + QK-Norm（替 Gemma 2 的 soft-cap）
位置编码：双 RoPE——局部层基频 10k / 全局层基频 1M；32K 预训练→128K 缩放 8 外推
规模/上下文：1B–27B（dense），128K（1B 为 32K）
多模态：冻结 SigLIP-400M，图像压成 256 soft token，Pan&Scan
Mamba / Jamba / RWKV（非 Transformer / 混合）

Mamba-2：选择性状态空间模型，等价于结构化线性注意力；O(L) 线性、固定状态、无 KV cache
Jamba：Mamba + Attention + MoE 混合，Mamba:Attention ≈ 7:1；1.5 Large 398B/94B，256K
RWKV-7：纯线性 RNN（WKV），训练并行、推理 O(1) 恒定显存、无位置编码（含衰减），1.5B–14B，无限上下文
OpenAI GPT-OSS

注意力：带状滑窗(128) + 全量交替（类 GPT-3）；GQA 组大小 8（64 Q 头 / 8 KV 头）；每头 softmax 分母带可学习 bias（attention sink）
FFN：MoE，120b 128 专家 / 20b 32 专家，均 top-4；门控 SwiGLU 含 clamp 钳制 + 残差
归一化：RMSNorm（Pre-LN）
位置编码：RoPE + YaRN，扩到 128K
精度：MoE 权重 MXFP4 QAT（4.25 bit）→ 120b 单卡 80GB、20b 仅 16GB
规模：120b 117B/5.1B · 20b 21B/3.6B；Apache 2.0
