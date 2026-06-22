# src/datasets/ifeng_processor.py
"""
凤凰新闻数据集处理模块
假设凤凰新闻数据格式为：
{
  "text": "主席在北京人民大会堂发表重要讲话。",
  "entities": [
    {"text": "主席", "type": "PER", "start": 0, "end": 3},
    {"text": "北京", "type": "LOC", "start": 4, "end": 6},
    {"text": "人民大会堂", "type": "ORG", "start": 6, "end": 11}
  ]
}
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from .base_processor import BaseDataProcessor

class IfengDataProcessor(BaseDataProcessor):
    """凤凰新闻数据集处理器"""
    
    def __init__(self, data_dir: str = "data/ifeng"):
        super().__init__(data_dir)
        self.entity_types = ['PER', 'ORG', 'LOC', 'GPE', 'TIME', 'EVENT']
    
    def convert_to_bio(self, input_file: str, output_file: str):
        """
        将凤凰新闻格式转换为标准BIO格式
        
        Args:
            input_file: 原始数据文件
            output_file: 输出BIO格式文件
        """
        with open(Path(self.data_dir) / input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        bio_samples = []
        for item in raw_data:
            text = item['text']
            entities = item.get('entities', [])
            
            # 中文按字分割
            tokens = list(text)
            ner_tags = ['O'] * len(tokens)
            
            # 处理实体标注
            for entity in entities:
                start = entity['start']
                end = entity['end']
                entity_type = entity['type']
                
                # 验证实体边界
                entity_text = entity.get('text', '')
                if entity_text and text[start:end] != entity_text:
                    print(f"警告: 实体文本不匹配: '{text[start:end]}' != '{entity_text}'")
                
                if 0 <= start < end <= len(tokens):
                    ner_tags[start] = f'B-{entity_type}'
                    for i in range(start + 1, end):
                        ner_tags[i] = f'I-{entity_type}'
            
            bio_samples.append({
                'tokens': tokens,
                'ner_tags': ner_tags,
                'text': text,
                'id': item.get('id', f'sample_{len(bio_samples)}')
            })
        
        # 保存转换后的数据
        with open(Path(self.data_dir) / output_file, 'w', encoding='utf-8') as f:
            json.dump(bio_samples, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成: {len(bio_samples)} 条样本已保存到 {output_file}")
    
    def analyze_dataset(self, split: str = 'train'):
        """分析数据集统计信息"""
        from collections import Counter
        
        file_path = Path(self.data_dir) / f'{split}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'num_samples': len(data),
            'text_lengths': [],
            'entity_counts': Counter(),
            'entity_lengths': [],
            'label_distribution': Counter()
        }
        
        for sample in data:
            # 文本长度统计
            stats['text_lengths'].append(len(sample['tokens']))
            
            # 标签分布
            stats['label_distribution'].update(sample['ner_tags'])
            
            # 实体统计
            current_entity = None
            entity_length = 0
            
            for token, tag in zip(sample['tokens'], sample['ner_tags']):
                if tag.startswith('B-'):
                    if current_entity:
                        stats['entity_counts'][current_entity] += 1
                        stats['entity_lengths'].append(entity_length)
                    current_entity = tag[2:]
                    entity_length = 1
                elif tag.startswith('I-'):
                    if current_entity and tag[2:] == current_entity:
                        entity_length += 1
                elif tag == 'O':
                    if current_entity:
                        stats['entity_counts'][current_entity] += 1
                        stats['entity_lengths'].append(entity_length)
                        current_entity = None
                        entity_length = 0
            
            # 处理最后一个实体
            if current_entity:
                stats['entity_counts'][current_entity] += 1
                stats['entity_lengths'].append(entity_length)
        
        return stats
