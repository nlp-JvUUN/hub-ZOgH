"""
数学问题数据集处理
Math Problem Dataset Handler
"""

import re
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class MathDataset(Dataset):
    """数学问题数据集"""
    
    def __init__(self, split: str = "train[:1000]", dataset_name: str = "gsm8k"):
        """
        初始化数学数据集
        
        Args:
            split: 数据集切分
            dataset_name: 数据集名称
        """
        self.dataset_name = dataset_name
        print(f"Loading dataset: {dataset_name}, split: {split}")
        
        # 加载GSM8K数据集
        if dataset_name == "gsm8k":
            self.dataset = load_dataset("gsm8k", "main", split=split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        print(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """获取单个样本"""
        item = self.dataset[idx]
        
        # GSM8K格式: question和answer
        question = item["question"]
        answer = item["answer"]
        
        # 提取最终数字答案
        final_answer = self.extract_answer(answer)
        
        return {
            "question": question,
            "full_answer": answer,
            "final_answer": final_answer,
            "idx": idx
        }
    
    @staticmethod
    def extract_answer(answer_text: str) -> str:
        """
        从答案文本中提取最终数字答案
        GSM8K格式通常是: 解题过程\n#### 答案
        """
        # 尝试提取#### 后的答案
        match = re.search(r'####\s*(.+)', answer_text)
        if match:
            return match.group(1).strip()
        
        # 如果没有####标记，尝试提取最后一个数字
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return answer_text.strip()
    
    @staticmethod
    def check_answer(predicted: str, ground_truth: str) -> bool:
        """
        检查预测答案是否正确
        
        Args:
            predicted: 预测答案
            ground_truth: 真实答案
            
        Returns:
            是否正确
        """
        # 提取数字并比较
        pred_num = MathDataset.extract_number(predicted)
        gt_num = MathDataset.extract_number(ground_truth)
        
        if pred_num is not None and gt_num is not None:
            # 数值比较（允许小误差）
            return abs(pred_num - gt_num) < 1e-6
        
        # 字符串比较（去除空格和逗号）
        pred_clean = re.sub(r'[\s,]', '', predicted.lower())
        gt_clean = re.sub(r'[\s,]', '', ground_truth.lower())
        
        return pred_clean == gt_clean
    
    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """从文本中提取数字"""
        # 移除逗号
        text = text.replace(',', '')
        
        # 尝试提取数字
        match = re.search(r'-?\d+(?:\.\d+)?', text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None


def format_prompt(question: str, include_answer: bool = False, answer: str = "") -> str:
    """
    格式化提示词
    
    Args:
        question: 问题文本
        include_answer: 是否包含答案
        answer: 答案文本
        
    Returns:
        格式化的提示词
    """
    prompt = f"""Solve this math problem step by step and provide the final answer.

Question: {question}

Please show your work and end with "Therefore, the answer is: [your final answer]"

Solution:"""
    
    if include_answer:
        prompt += f"\n{answer}"
    
    return prompt


def parse_model_output(output: str) -> str:
    """
    从模型输出中解析最终答案
    
    Args:
        output: 模型生成的完整输出
        
    Returns:
        提取的最终答案
    """
    # 尝试提取"Therefore, the answer is:"后的内容
    match = re.search(r'Therefore,?\s+the answer is:?\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 移除可能的标点符号
        answer = re.sub(r'[.,;]$', '', answer)
        return answer
    
    # 尝试提取#### 后的答案
    match = re.search(r'####\s*(.+)', output)
    if match:
        return match.group(1).strip()
    
    # 尝试提取最后一个数字
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', output)
    if numbers:
        return numbers[-1]
    
    # 返回最后一行（去除空白）
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return output.strip()


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批处理函数
    
    Args:
        batch: 批次数据
        
    Returns:
        整理后的批次
    """
    return {
        "questions": [item["question"] for item in batch],
        "full_answers": [item["full_answer"] for item in batch],
        "final_answers": [item["final_answer"] for item in batch],
        "indices": [item["idx"] for item in batch]
    }
