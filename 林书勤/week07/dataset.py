import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class NERDatasetConfig:
    """NER数据集配置类"""
    name: str  # 数据集名称，如'peoples_daily'、'ifeng'等
    entity_types: List[str]  # 实体类型列表，如['PER', 'ORG', 'LOC']
    format_type: str  # 数据格式：'bio'、'json'、'conll'等
    label2id: Dict[str, int]  # 标签到ID的映射
    id2label: Dict[int, str]  # ID到标签的映射
    data_dir: str  # 数据目录路径

class NERDatasetLoader:
    """通用NER数据集加载器"""
    
    def __init__(self, dataset_name: str = 'peoples_daily', data_dir: str = None):
        """
        初始化数据集加载器
        
        Args:
            dataset_name: 数据集名称
                - 'peoples_daily': 人民日报标准格式
                - 'ifeng': 凤凰新闻自定义格式
                - 'cluener': CLUENER细粒度NER
                - 'weibo': 微博NER数据集
                - 'resume': 中文简历NER
                - 'ontonotes': OntoNotes中文
            data_dir: 数据目录路径
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir else Path('data') / dataset_name
        
        # 根据不同数据集配置参数
        self.config = self._get_dataset_config()
        
    def _get_dataset_config(self) -> NERDatasetConfig:
        """获取数据集配置"""
        if self.dataset_name == 'peoples_daily':
            entity_types = ['PER', 'ORG', 'LOC']
            format_type = 'bio'
        elif self.dataset_name == 'ifeng':
            entity_types = ['PER', 'ORG', 'LOC', 'GPE', 'TIME']  # 示例，可根据实际调整
            format_type = 'json'
        elif self.dataset_name == 'cluener':
            entity_types = ['address', 'book', 'company', 'game', 'government',
                          'movie', 'name', 'organization', 'position', 'scene']
            format_type = 'cluener'
        elif self.dataset_name == 'weibo':
            entity_types = ['PER', 'ORG', 'LOC', 'GPE']
            format_type = 'weibo'
        elif self.dataset_name == 'resume':
            entity_types = ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE']
            format_type = 'resume'
        elif self.dataset_name == 'ontonotes':
            entity_types = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 
                          'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 
                          'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 
                          'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
            format_type = 'conll'
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        # 生成BIO标签
        label_list = ['O']
        for entity in entity_types:
            label_list.extend([f'B-{entity}', f'I-{entity}'])
        
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        return NERDatasetConfig(
            name=self.dataset_name,
            entity_types=entity_types,
            format_type=format_type,
            label2id=label2id,
            id2label=id2label,
            data_dir=str(self.data_dir)
        )
    
    def load_data(self, split: str = 'train'):
        """
        加载指定数据集
        
        Args:
            split: 数据分割，'train'、'validation'、'test'
            
        Returns:
            List[Dict]: 包含tokens和ner_tags的样本列表
        """
        data_path = self.data_dir / f'{split}.json'
        
        if not data_path.exists():
            # 尝试其他可能的文件格式
            possible_paths = [
                self.data_dir / f'{split}.json',
                self.data_dir / f'{split}.jsonl',
                self.data_dir / f'{split}.txt',
                self.data_dir / f'{split}.conll',
            ]
            
            for path in possible_paths:
                if path.exists():
                    data_path = path
                    break
            else:
                raise FileNotFoundError(f"找不到{split}数据文件，请检查路径: {self.data_dir}")
        
        if self.config.format_type == 'bio':
            return self._load_bio_format(data_path)
        elif self.config.format_type == 'json':
            return self._load_json_format(data_path)
        elif self.config.format_type == 'cluener':
            return self._load_cluener_format(data_path)
        elif self.config.format_type == 'conll':
            return self._load_conll_format(data_path)
        else:
            return self._load_auto_format(data_path)
    
    def _load_bio_format(self, file_path: Path) -> List[Dict]:
        """加载BIO格式数据（人民日报格式）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            if 'tokens' in item and 'ner_tags' in item:
                # 确保标签是字符串格式
                ner_tags = [self.config.id2label.get(tag, tag) if isinstance(tag, int) else tag 
                           for tag in item['ner_tags']]
                samples.append({
                    'tokens': item['tokens'],
                    'ner_tags': ner_tags,
                    'text': ''.join(item['tokens']) if 'text' not in item else item['text']
                })
        return samples
    
    def _load_json_format(self, file_path: Path) -> List[Dict]:
        """加载自定义JSON格式（适合凤凰新闻等）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            if 'text' in item and 'entities' in item:
                # 从文本和实体标注生成BIO标签
                text = item['text']
                entities = item['entities']
                
                # 将文本转为字符列表（中文按字分割）
                tokens = list(text)
                ner_tags = ['O'] * len(tokens)
                
                for entity in entities:
                    start = entity.get('start_idx', entity.get('start', 0))
                    end = entity.get('end_idx', entity.get('end', 0))
                    entity_type = entity.get('type', entity.get('label', 'UNK'))
                    
                    if 0 <= start < end <= len(tokens):
                        # 分配BIO标签
                        ner_tags[start] = f'B-{entity_type}'
                        for i in range(start + 1, end):
                            ner_tags[i] = f'I-{entity_type}'
                
                samples.append({
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                    'text': text,
                    'raw_entities': entities
                })
        return samples
