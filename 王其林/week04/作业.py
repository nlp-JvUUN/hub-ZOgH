import torch
import torch.nn as nn
from transformers import BertModel

# ─── 超参数 ───────────────────
SEED        = 42
SEQ_LEN     = 16
BATCH_SIZE  = 1

torch.manual_seed(SEED)

"""
实现自定义transformer层
"""
class DIYTransformerLayer():
    def __init__(self, 
                 hidden_size, # 特征数
                 num_heads, # 多头数量
                 q_w, q_b,  # (hidden_size, hidden_size)
                 k_w, k_b,  # (hidden_size, hidden_size)
                 v_w, v_b,  # (hidden_size, hidden_size)
                 attention_out_w, attention_out_b,  # (hidden_size, hidden_size)
                 attention_layer_norm_w, attention_layer_norm_b,    # (hidden_size)
                 intermediate_w, intermediate_b, # (intermediate_size, hidden_size)
                 output_w, output_b,    # (hidden_size, intermediate_size)
                 ff_layer_norm_w, ff_layer_norm_b): # (hidden_size)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_w = q_w
        self.q_b = q_b
        self.k_w = k_w
        self.k_b = k_b
        self.v_w = v_w
        self.v_b = v_b
        self.attention_out_w = attention_out_w
        self.attention_out_b = attention_out_b
        self.attention_layer_norm_w = attention_layer_norm_w
        self.attention_layer_norm_b = attention_layer_norm_b
        self.intermediate_w = intermediate_w
        self.intermediate_b = intermediate_b
        self.output_w = output_w
        self.output_b = output_b
        self.ff_layer_norm_w = ff_layer_norm_w
        self.ff_layer_norm_b = ff_layer_norm_b

    def softmax(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

    def attention_forward(self, x): # (batch_size, max_len, hidden_size)
        """多头自注意力计算"""
        batch_size, max_len, hidden_size = x.shape
        q = x @ self.q_w.T + self.q_b     # (batch_size, max_len, hidden_size)
        k = x @ self.k_w.T + self.k_b     # (batch_size, max_len, hidden_size)
        v = x @ self.v_w.T + self.v_b     # (batch_size, max_len, hidden_size)
        head_size = int(self.hidden_size / self.num_heads)
        q = q.reshape(batch_size, max_len, self.num_heads, head_size).transpose(1, 2)   # (batch_size, num_heads, max_len, head_size)
        k = k.reshape(batch_size, max_len, self.num_heads, head_size).transpose(1, 2)   # (batch_size, num_heads, max_len, head_size)
        v = v.reshape(batch_size, max_len, self.num_heads, head_size).transpose(1, 2)   # (batch_size, num_heads, max_len, head_size)
        qk = q @ k.transpose(2, 3)  # (batch_size, num_heads, max_len, max_len)
        qk /= torch.sqrt(torch.tensor(head_size, dtype=torch.long)) 
        qk = self.softmax(qk)
        qkv = qk @ v    # (batch_size, num_heads, max_len, head_size)
        qkv = qkv.transpose(1, 2).reshape(batch_size, max_len, self.hidden_size)     # (batch_size, max_len, hidden_size)
        attention = qkv @ self.attention_out_w.T + self.attention_out_b
        return attention
    
    def feed_forward(self, x):  # (batch_size, max_len, hidden_size)
        """前馈网络"""
        x = x @ self.intermediate_w.T + self.intermediate_b   # (batch_size, max_len, intermediate_size)
        x = nn.functional.gelu(x)
        x = x @ self.output_w.T + self.output_b   # (batch_size, max_len, hidden_size)
        return x
    
    def layer_norm(self, x, w, b):
        """归一化"""
        eps = 1e-5
        avg = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        normal_x = (x - avg) / torch.sqrt(var + eps)
        out = normal_x * w + b
        return out

    def forward(self, x): # (batch_size, max_len, hidden_size)
        # 多头自注意力
        attention = self.attention_forward(x)  # (batch_size, max_len, hidden_size)
        # 残差连接 归一化
        attention_norm_x = self.layer_norm(x + attention, self.attention_layer_norm_w, self.attention_layer_norm_b)
        # 前馈网络
        ff_output = self.feed_forward(attention_norm_x)
        # 残差连接 归一化
        out = self.layer_norm(attention_norm_x + ff_output, self.ff_layer_norm_w, self.ff_layer_norm_b)
        return out
    

def main():
    # 加载官方模型
    model = BertModel.from_pretrained(r"/Users/wangqilin/个人/学习/week4 语言模型/bert-base-chinese", return_dict=False)
    bert_layer = model.encoder.layer[0]
    config = model.config

    # 提取参数
    params = {
        "q_w": bert_layer.attention.self.query.weight,
        "q_b": bert_layer.attention.self.query.bias,
        "k_w": bert_layer.attention.self.key.weight,
        "k_b": bert_layer.attention.self.key.bias,
        "v_w": bert_layer.attention.self.value.weight,
        "v_b": bert_layer.attention.self.value.bias,
        "attention_out_w": bert_layer.attention.output.dense.weight,
        "attention_out_b": bert_layer.attention.output.dense.bias,
        "attention_layer_norm_w": bert_layer.attention.output.LayerNorm.weight,
        "attention_layer_norm_b": bert_layer.attention.output.LayerNorm.bias,
        "intermediate_w": bert_layer.intermediate.dense.weight,
        "intermediate_b": bert_layer.intermediate.dense.bias,
        "output_w": bert_layer.output.dense.weight,
        "output_b": bert_layer.output.dense.bias,
        "ff_layer_norm_w": bert_layer.output.LayerNorm.weight,
        "ff_layer_norm_b": bert_layer.output.LayerNorm.bias,
    }

    # 实例化自定义transformerLayer
    diy = DIYTransformerLayer(
        config.hidden_size,
        config.num_attention_heads,
        **params
    )

    # 随机输入 (batch_size, seq_len, hidden_size)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, config.hidden_size)
    
    # 官方模型推理
    bert_layer.eval()
    with torch.no_grad():
        official_output = bert_layer(input_tensor, attention_mask=None, return_dict=False)[0]  # (batch_size, seq_len, hidden_size)

    # 自定义模型推理
    diy_output = diy.forward(input_tensor)   # (batch_size, seq_len, hidden)

    # 输出结果比较
    print(official_output)
    print(diy_output)

if __name__ == "__main__":
    main()
