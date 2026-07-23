"""下载 BGE 模型到本地目录（使用镜像源）"""
from transformers import AutoModel, AutoTokenizer
import os

# 使用 HuggingFace 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = 'BAAI/bge-small-zh-v1.5'
save_path = '/Users/wangxinyu/Desktop/python/最新/pretrain_models/bge-small-zh-v1.5'

print(f'开始下载 {model_name}...')
print('使用镜像源: https://hf-mirror.com')
print('这可能需要几分钟，请耐心等待...')

try:
    # 下载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # 保存到本地
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f'\n模型已保存到: {save_path}')
    print('文件列表:')
    total_size = 0
    for f in sorted(os.listdir(save_path)):
        size = os.path.getsize(os.path.join(save_path, f))
        total_size += size
        print(f'  {f}: {size/(1024*1024):.2f} MB')
    print(f'总大小: {total_size/(1024*1024):.2f} MB')
    print('\n下载成功！')
    
except Exception as e:
    print(f'下载失败: {e}')
    print('\n请检查网络连接，或手动从 https://hf-mirror.com 下载')
