import torch 
import torch.nn as nn
import math
class transformer(nn.Module):
    #为什么不用输入input_size：transformer的输入是[batch_size, seq_len, hidden_size]
    # num_attention_heads: 多头注意力的头数
    # intermediate_size: 前馈网络中间层维度（FFN)
    # dropout: 丢弃概率
    def __init__(self,hidden_size=768,num_attention_heads=12,intermediate_size=3072,dropout=0.1):
        super(transformer,self).__init__()
        self.hidden_size=hidden_size
        self.num_attention_heads=num_attention_heads
        self.attention_head_size=int(hidden_size/num_attention_heads)
#相关的qkv hidden_size决定了可学习参数的数量
        self.query=nn.Linear(hidden_size,hidden_size)
        self.key=nn.Linear(hidden_size,hidden_size)
        self.value=nn.Linear(hidden_size,hidden_size)
        self.attention_output=nn.Linear(hidden_size,hidden_size)
#y=x*w^T+b
        self.intermediate_dense=nn.Linear(hidden_size,intermediate_size)#算完拿到结果传给gelu,再传给
        self.output_dense=nn.Linear(intermediate_size,hidden_size)
#归一化层 残差链接在forward中实现不在__init__中定义
        self.attention_layer_norm=nn.LayerNorm(hidden_size,eps=1e-12)
        self.ffn_layer_norm=nn.LayerNorm(hidden_size,eps=1e-12)

        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(dropout)
    def transpose_for_scores(self,x):
        batch_size,seq_len,_=x.size()
        x=x.view(batch_size,seq_len,self.num_attention_heads,self.attention_head_size)
        return x.permute(0,2,1,3)
    def forward(self,hidden_states):
        #线性投影生成qkv
        query_layer=self.transpose_for_scores(self.query(hidden_states))
        key_layer=self.transpose_for_scores(self.key(hidden_states))
        value_layer=self.transpose_for_scores(self.value(hidden_states))
        #计算注意力分数 q*k^t
        attention_scores=torch.matmul(query_layer,key_layer.transpose(-1,-2))
        #缩放注意力分数
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)
        #加权求和
        context_layer=torch.matmul(attention_probs,value_layer)
        #合并多头 view方法要求张量在内存中是连续存储的,可以换成.reshape方法
        context_layer=context_layer.permute(0,2,1,3)
        new_context_layer_shape=context_layer.size()[:-2]+(self.hidden_size,)
        context_layer=context_layer.reshape(*new_context_layer_shape)

        attention_output=self.attention_output(context_layer)
        attention_output=self.dropout(attention_output)
        #残差链接+LayerNorm
        attention_output=self.attention_layer_norm(attention_output+hidden_states)
        #升维 激活 降维 丢弃
        intermediate_output=self.intermediate_dense(attention_output)
        intermediate_output=self.gelu(intermediate_output)

        ffn_output=self.output_dense(intermediate_output)
        ffn_output=self.dropout(ffn_output)
        layer_output=self.ffn_layer_norm(ffn_output+attention_output)
        return layer_output
if __name__=="__main__":
    batch_size=2
    seq_len=5
    hidden_size=768
    input_tensor=torch.randn(batch_size,seq_len,hidden_size)
    transformer_layer=transformer(hidden_size=hidden_size)
    output_tensor=transformer_layer(input_tensor)
    print(f"输入形状:{input_tensor.shape}")
    print(f"输入形状:{output_tensor.shape}")
   
