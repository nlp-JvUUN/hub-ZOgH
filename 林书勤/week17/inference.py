"""
推理脚本 - 使用训练好的GRPO模型
Inference Script for Trained GRPO Model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_dataset import format_prompt, parse_model_output, MathDataset


class MathSolver:
    """数学问题求解器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化求解器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def solve(
        self,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        show_work: bool = True
    ) -> dict:
        """
        求解数学问题
        
        Args:
            question: 问题文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            show_work: 是否显示解题过程
            
        Returns:
            包含答案和解题过程的字典
        """
        # 格式化提示
        prompt = format_prompt(question)
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析答案
        final_answer = parse_model_output(generated_text)
        
        result = {
            "question": question,
            "solution": generated_text,
            "final_answer": final_answer
        }
        
        if show_work:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"\nSolution:\n{generated_text}")
            print(f"\nFinal Answer: {final_answer}")
            print(f"{'='*60}\n")
        
        return result
    
    def batch_solve(self, questions: list, show_progress: bool = True) -> list:
        """
        批量求解问题
        
        Args:
            questions: 问题列表
            show_progress: 是否显示进度
            
        Returns:
            结果列表
        """
        results = []
        
        for i, question in enumerate(questions):
            if show_progress:
                print(f"\nSolving problem {i+1}/{len(questions)}...")
            
            result = self.solve(question, show_work=show_progress)
            results.append(result)
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用GRPO模型求解数学问题")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./grpo_math_model/final",
        help="模型路径"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="要求解的问题"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )
    
    args = parser.parse_args()
    
    # 创建求解器
    solver = MathSolver(args.model_path, args.device)
    
    if args.question:
        # 求解单个问题
        solver.solve(args.question, temperature=args.temperature)
    else:
        # 交互模式
        print("\n" + "="*60)
        print("GRPO Math Solver - Interactive Mode")
        print("="*60)
        print("Enter 'quit' or 'exit' to stop")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\nEnter your math question: ").strip()
                
                if question.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                solver.solve(question, temperature=args.temperature)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
