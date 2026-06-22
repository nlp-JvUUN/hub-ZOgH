import argparse
from dataset import NERDatasetLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='peoples_daily',
                       choices=['peoples_daily', 'ifeng', 'cluener', 'weibo', 'resume', 'ontonotes'],
                       help='选择数据集')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='自定义数据目录路径')
    parser.add_argument('--num_train', type=int, default=8000,
                       help='训练样本数（用于快速实验）')
    parser.add_argument('--use_crf', action='store_true',
                       help='使用CRF层')
    # ... 其他参数
    
    args = parser.parse_args()
    
    # 加载数据集
    dataset_loader = NERDatasetLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir
    )
    
    # 加载训练、验证、测试数据
    train_data = dataset_loader.load_data('train')
    val_data = dataset_loader.load_data('validation')
    test_data = dataset_loader.load_data('test')
    
    # 获取标签映射
    label2id = dataset_loader.config.label2id
    id2label = dataset_loader.config.id2label
    
    print(f"数据集: {args.dataset}")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"标签数量: {len(label2id)}")
    print(f"实体类型: {dataset_loader.config.entity_types}")
    
    # 后续训练代码保持不变...
