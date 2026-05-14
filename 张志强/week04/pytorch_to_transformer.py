import math

import torch
import torch.nn as nn
'''
尝试用pytorch实现自己对bert模型结构的理解 embedding-->transformer
'''
test_word = '我爱北京天安门'
# 1. 构建一个词表
vocab = {'<PAD>': 0, '<UNK>': 1, "[cls]": 2, "[sep]": 3, }
for word in test_word:
    if word not in vocab:
        vocab[word] = len(vocab)
# {'<PAD>': 0, '<UNK>': 1, '[cls]': 2, '[sep]': 3, '我': 4, '爱': 5, '北': 6, '京': 7, '天': 8, '安': 9, '门': 10}
print(f"词表:{vocab}")
# 2. 添加 CLS 和 SEP
word_list = list(test_word)
word_list.insert(0, "[cls]")
word_list.append("[sep]")
print(f"word_list:{word_list}")

# 3. token 转 id
word_list_id = [vocab[word] for word in word_list]
# word_list_id:[2, 4, 5, 6, 7, 8, 9, 10, 3]
print(f"word_list_id:{word_list_id}")
# 4. position ids
position_ids = [i for i in range(len(word_list_id))]
print(f"position_ids:{position_ids}")
# 5. segment ids 根据词表构建segment  简单处理  因为只有一句  所以都是0
segment_ids = [0 for i in range(len(word_list_id))]
print(f"segment_ids:{segment_ids}")

# 6. 转 Tensor的张量
tensor_token = torch.LongTensor(word_list_id)
tensor_pos = torch.LongTensor(position_ids)
tensor_seg = torch.LongTensor(segment_ids)
print(f"tensor_token:{tensor_token}")
print(f"tensor_pos:{tensor_pos}")
print(f"tensor_seg:{tensor_seg}")

# 7. 构建bert的三层  embedding
EMBED_DIM = 768
vocab_size = len(vocab)
# token 普通的Embedding层
embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
# position 层
position_embedding = nn.Embedding(512, EMBED_DIM)
# segment 层 区分A/B句
segment_embedding = nn.Embedding(2, EMBED_DIM)

token_emb = embedding(tensor_token)
pos_emb = position_embedding(tensor_pos)
seg_emb = segment_embedding(tensor_seg)
# 三者相加
output = token_emb + seg_emb + pos_emb
#将 output 归一化后输出
LayerNorm = nn.LayerNorm(EMBED_DIM)
embedding_output = LayerNorm(output) #输出不变还是 [9, 768]
print(embedding_output)
print(embedding_output.size())  #torch.Size([9, 768])

#8. 多头 Q K V 的线性层 每个都是768 768
W_Q = nn.Linear(EMBED_DIM, EMBED_DIM)
W_K = nn.Linear(EMBED_DIM, EMBED_DIM)
W_V = nn.Linear(EMBED_DIM, EMBED_DIM)

# 输入:
Q = W_Q(embedding_output)
K = W_K(embedding_output)
V = W_V(embedding_output)
print(f"Q shape:{Q.shape}")

# 9. 拆分多个头
seq_len = Q.shape[0]
print(f"seq_len:{seq_len}")
# 拆分为12份
NUM_HEADS = 12
#每个拆分后的维度是64
HEAD_DIM = 64


# 多头机制
def transpose_for_scores(x, attention_head_size, num_attention_heads):
    # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
    max_len, hidden_size = x.shape
    x = x.reshape(max_len, num_attention_heads, attention_head_size)
    x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
    return x
# [seq_len, 768] 到 [seq_len, 12, 64]
Q = transpose_for_scores(Q,HEAD_DIM,NUM_HEADS)

K = transpose_for_scores(K,HEAD_DIM,NUM_HEADS)

V = transpose_for_scores(V,HEAD_DIM,NUM_HEADS)
#torch.Size([12, 9, 64])
print(f"拆分后Q shape:{Q.shape}")
print(f"拆分后K shape:{K.shape}")
print(f"拆分后V shape:{V.shape}")
# 10. 计算 注意力  得到Score
# K.transpose(-1, -2) 是将K的维度倒数第二个和倒数第一个进行转置  为了达到矩阵乘法的要求 [9,64] [64,9]  最后[9,9]
attention_scores = torch.matmul(
    Q,
    K.transpose(-1, -2)
)
print(f"Scores shape:{attention_scores.shape}")
# 得到Scores后在除以根号下的dk
attention_scores = attention_scores / math.sqrt(HEAD_DIM)
# 11. Softmax 计算
softmax_result = torch.softmax(attention_scores, dim=-1)
print(f"softmax_result shape:{softmax_result.shape}")
# 12. softmax_result × V
qkv = torch.matmul(softmax_result, V)
print(f"qkv shape:{qkv.shape}")
# 13. 开始多头拼接起来
qkv = qkv.permute(1, 0, 2).reshape(-1, EMBED_DIM)
print(f"拼接后 qkv shape:{qkv.shape}")
# 14. 在经过线性层输出
attention_output = nn.Linear(EMBED_DIM,EMBED_DIM)(qkv)
print(f"attention_output shape:{attention_output.shape}")
# 15. 残差连接 + 层归一化
# 残差:
attention_output = (
      embedding_output +attention_output
)
#层归一化
attention_output = nn.LayerNorm(EMBED_DIM)(attention_output)

# 16. 前馈网络  两个线性层+一个GELU激活函数
def feed_forward(x):
   x_linear =  nn.Linear(EMBED_DIM,3072)(x)
   gelu_result = nn.GELU()(x_linear)
   return nn.Linear(3072,EMBED_DIM)(gelu_result)

ffn_output = feed_forward(attention_output)
print(f"FFN输出 shape:{ffn_output.shape}")
# 17. 在经过一层 残差 + LayerNorm
ffn_output = (
      ffn_output +attention_output
)
#层归一化
final_output = nn.LayerNorm(EMBED_DIM)(ffn_output)
print(f"最终Transformer输出:{final_output}")
print(f"最终shape:{final_output.shape}")
