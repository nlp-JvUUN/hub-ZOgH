"""
统一的数据预处理脚本
支持多种中文NER数据集的格式转换
"""

import argparse
import json
from pathlib import Path
from datasets.ifeng_processor import IfengDataProcessor
from datasets.cluener_processor import CluenerProcessor
from datasets.weibo_processor import WeiboProcessor

def preprocess_ifeng(raw_dir: str, output_dir: str):
    """预处理凤凰新闻数据集"""
    processor = IfengDataProcessor(raw_dir)
    
    # 假设原始文件结构
    # raw_dir/
    #   train_raw.json
    #   dev_raw.json
    #   test_raw.json
    
    print("处理凤凰新闻数据集...")
    
    # 转换各分割集
    for split in ['train', 'dev', 'test']:
        input_file = f"{split}_raw.json"
        output_file = f"{split}.json"
        
        if (Path(raw_dir) / input_file).exists():
            processor.convert_to_bio(input_file, output_file)
            
            # 分析数据集
            stats = processor.analyze_dataset(split)
            print(f"\n{split}集统计:")
            print(f"  样本数: {stats['num_samples']}")
            print(f"  平均文本长度: {sum(stats['text_lengths'])/len(stats['text_lengths']):.1f}")
            print(f"  实体分布: {dict(stats['entity_counts'])}")
            print(f"  平均实体长度: {sum(stats['entity_lengths'])/len(stats['entity_lengths']):.1f}")
    
    print(f"\n处理完成！数据已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="NER数据集预处理工具")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ifeng', 'cluener', 'weibo', 'resume', 'custom'],
                       help='要预处理的数据集类型')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='输出目录')
    parser.add_argument('--format', type=str, default='bio',
                       choices=['bio', 'json', 'conll'],
                       help='输出格式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output_dir) / args.dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'ifeng':
        preprocess_ifeng(args.input_dir, str(output_path))
    elif args.dataset == 'cluener':
        # 处理CLUENER数据集
        pass
    elif args.dataset == 'weibo':
        # 处理微博数据集
        pass
    else:
        print(f"暂不支持 {args.dataset} 数据集的自动预处理")
        print("请参考 ifeng_processor.py 实现自定义处理逻辑")

if __name__ == '__main__':
    main()
