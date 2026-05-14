import torch
import math


class PureTransformerModel:
    def __init__(self, vocab_size, max_position_embeddings, num_layers, embed_dim, num_heads, ff_dim, num_classes):
        """
        __init__ 初始化参数：仅定义结构超参数
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes

        # 内部权重字典，用于存放通过 load_weights 读入的所有底层 Tensor 参数
        self.weights = {}

    def load_weights(self, weights_dict):
        """
        load_weights 获取参数：接收一个包含纯 Tensor 的权重字典
        """
        self.weights = weights_dict
        print("--- 所有权重参数成功加载至内存 ---")

    # 核心算子部分（替代 nn.LayerNorm 和 nn.Linear）

    def _layer_norm(self, x, gamma, beta, eps=1e-12):
        """ LayerNorm：针对最后一维特征轴做归一化"""
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return gamma * x_norm + beta

    def _dense(self, x, weight, bias):
        """ 全连接层：y = x @ W^T + b"""
        return x @ weight.t() + bias

    # FORWARD 前向传播主干与子模块

    def get_embedding(self, input_ids, position_ids):
        """ get_embedding: 替换词句在词表中的具体位置，提取稠密特征"""
        # 利用 input_ids 作为索引从大矩阵里抽取行向量
        token_embeds = self.weights['embedding.token_weights'][input_ids]
        pos_embeds = self.weights['embedding.position_weights'][position_ids]
        return token_embeds, pos_embeds

    def embedding_forward(self, input_ids):
        """ embedding_forward: TOKEN序列相加，再归一化"""
        batch_size, seq_len = input_ids.shape

        # 动态动态创建 position_ids: [1, seq_len] 并广播
        position_ids = torch.arange(seq_len, dtype=torch.long).expand(batch_size, seq_len)

        # 获取两组嵌入特征
        token_embeds, pos_embeds = self.get_embedding(input_ids, position_ids)

        # 序列特征直接相加
        embedding_output = token_embeds + pos_embeds

        # 经过 Embedding 层的 LayerNorm 归一化
        embedding_output = self._layer_norm(
            embedding_output,
            self.weights['embedding.norm_gamma'],
            self.weights['embedding.norm_beta']
        )
        return embedding_output

    def transpose_for_scores(self, x):
        """transpose_for_scores: 多头注意力机制的维度切分与重排"""
        batch_size, seq_len, embed_dim = x.shape
        # 拆分特征维：[B, S, D] -> [B, S, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 维度置换调音：[B, num_heads, S, head_dim] 将 Head 维提成乘客
        return x.permute(0, 2, 1, 3)

    def self_attention(self, x, layer_idx):
        """self_attention 子层"""
        batch_size, seq_len, _ = x.shape
        prefix = f"transformerLayer.{layer_idx}."

        # 投影生成 Q, K, V
        q = self._dense(x, self.weights[prefix + 'q_w'], self.weights[prefix + 'q_b'])
        k = self._dense(x, self.weights[prefix + 'k_w'], self.weights[prefix + 'k_b'])
        v = self._dense(x, self.weights[prefix + 'v_w'], self.weights[prefix + 'v_b'])

        # 调用多头转换
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 矩阵乘法算两两关联度: [B, H, S, head_dim] @ [B, H, head_dim, S] -> [B, H, S, S]
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Softmax 归一化权重
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True)[0])
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        # 加权求和融合 V 特征: [B, H, S, S] @ [B, H, S, head_dim] -> [B, H, S, head_dim]
        context = torch.matmul(attn_weights, v)

        # 回缩多头形状: [B, H, S, head_dim] -> [B, S, H, head_dim] -> [B, S, D]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)

        # 注意力输出线性融合
        return self._dense(context, self.weights[prefix + 'attn_out_w'], self.weights[prefix + 'attn_out_b'])

    def feed_forward(self, x, layer_idx):
        """feed_forward 前馈网络子层"""
        prefix = f"transformerLayer.{layer_idx}."
        # 两层全连接，中间夹一个基于 torch.gelu 或 torch.relu 的非线性激活
        hidden = torch.relu(self._dense(x, self.weights[prefix + 'ffn1_w'], self.weights[prefix + 'ffn1_b']))
        return self._dense(hidden, self.weights[prefix + 'ffn2_w'], self.weights[prefix + 'ffn2_b'])

    def single_transformer_layer_forward(self, x, layer_idx):
        """single_transformer_layer_forward: 一层的transformer包含残差连接1和2"""
        prefix = f"transformerLayer.{layer_idx}."

        # --- 子阶段 1: Self-Attention + 残差连接 1 ---
        attn_out = self.self_attention(x, layer_idx)
        x = self._layer_norm(x + attn_out, self.weights[prefix + 'norm1_g'], self.weights[prefix + 'norm1_b'])  # 残差连接1

        # --- 子阶段 2: Feed-Forward + 残差连接 2 ---
        ffn_out = self.feed_forward(x, layer_idx)
        x = self._layer_norm(x + ffn_out, self.weights[prefix + 'norm2_g'], self.weights[prefix + 'norm2_b'])  # 残差连接2

        return x

    def all_transformer_layer_forward(self, x):
        """all_transformer_layer_forward: 执行L层的transformer循环"""
        for l in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, layer_idx=l)
        return x

    def pooler_output_layer(self, sequence_output):
        """pooler_output_layer后置: 最终输出分类向量：dot计算+tanh激活"""
        # 切片只抽取句子第 0 个位置（[CLS] Token）的向量作为整句代表
        cls_vector = sequence_output[:, 0, :]  # 形状从 [B, S, D] 降维降阶为 [B, D]

        # 执行 dot 密集矩阵计算变换
        pooled = self._dense(cls_vector, self.weights['pooler.weight'], self.weights['pooler.bias'])

        # tanh 激活函数收尾，返回固定表达
        pooled_output = torch.tanh(pooled)

        # 映射到最终类别
        logits = self._dense(pooled_output, self.weights['classifier.weight'], self.weights['classifier.bias'])
        return pooled_output

    def forward(self, input_ids):
        """主入口 forward"""
        # Step 1: 走 Embedding 流程
        x = self.embedding_forward(input_ids)
        # Step 2: 走 L 层循环的 Transformer 编码体
        x = self.all_transformer_layer_forward(x)
        # Step 3: 后置 Pooler 分类输出
        output = self.pooler_output_layer(x)
        return output


# Tensor 字典创建与测试运行

# 结构超参数定义
V, P, L, D, H, F, C_out = 1000, 20, 2, 768, 12, 3072, 2  # 2层Transformer，768特征

# 模拟人工手动制造底层权重参数字典（模拟外部 load 进来的模型文件）
raw_weights = {
    'embedding.token_weights': torch.randn(V, D),
    'embedding.position_weights': torch.randn(P, D),
    'embedding.norm_gamma': torch.ones(D), 'embedding.norm_beta': torch.zeros(D),
    'pooler.weight': torch.randn(D, D), 'pooler.bias': torch.randn(D),
    'classifier.weight': torch.randn(C_out, D), 'classifier.bias': torch.randn(C_out)
}
# 循环填充 2 层 Transformer 所需的所有矩阵和偏置
for l in range(L):
    p = f"transformerLayer.{l}."
    raw_weights.update({
        p + 'q_w': torch.randn(D, D), p + 'q_b': torch.randn(D),
        p + 'k_w': torch.randn(D, D), p + 'k_b': torch.randn(D),
        p + 'v_w': torch.randn(D, D), p + 'v_b': torch.randn(D),
        p + 'attn_out_w': torch.randn(D, D), p + 'attn_out_b': torch.randn(D),
        p + 'ffn1_w': torch.randn(F, D), p + 'ffn1_b': torch.randn(F),
        p + 'ffn2_w': torch.randn(D, F), p + 'ffn2_b': torch.randn(D),
        p + 'norm1_g': torch.ones(D), p + 'norm1_b': torch.zeros(D),
        p + 'norm2_g': torch.ones(D), p + 'norm2_b': torch.zeros(D),
    })

# 实例化并运行
model = PureTransformerModel(vocab_size=V, max_position_embeddings=P, num_layers=L, embed_dim=D, num_heads=H, ff_dim=F,
                             num_classes=C_out)
model.load_weights(raw_weights)

# 模拟输入：Batch=2，每句长度=5 的句子索引
dummy_input = torch.randint(0, V, (2, 5))
final_logits = model.forward(dummy_input)

print("输入文本 Token 序列形状:", dummy_input.shape)
print("最终经过 Pooler 计算输出的分类 Logits 形状:", final_logits.shape)
