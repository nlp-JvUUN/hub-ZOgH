"""
组件测试脚本
Test script for components
"""

import sys


def test_imports():
    """测试所有模块是否能正确导入"""
    print("Testing imports...")
    
    try:
        from config import GRPOConfig
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import config.py: {e}")
        return False
    
    try:
        from math_dataset import (
            MathDataset,
            format_prompt,
            parse_model_output,
            collate_fn
        )
        print("✓ math_dataset.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import math_dataset.py: {e}")
        return False
    
    try:
        from grpo_trainer import GRPOTrainer
        print("✓ grpo_trainer.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import grpo_trainer.py: {e}")
        return False
    
    return True


def test_config():
    """测试配置"""
    print("\nTesting configuration...")
    
    try:
        from config import GRPOConfig
        
        config = GRPOConfig()
        
        assert config.batch_size > 0, "Batch size must be positive"
        assert config.learning_rate > 0, "Learning rate must be positive"
        assert config.group_size > 0, "Group size must be positive"
        assert config.kl_coef >= 0, "KL coefficient must be non-negative"
        
        print(f"✓ Configuration validated")
        print(f"  - Model: {config.model_name}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Group size: {config.group_size}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - KL coefficient: {config.kl_coef}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_dataset_functions():
    """测试数据集处理函数"""
    print("\nTesting dataset functions...")
    
    try:
        from math_dataset import (
            format_prompt,
            parse_model_output,
            MathDataset
        )
        
        # 测试format_prompt
        question = "What is 2 + 2?"
        prompt = format_prompt(question)
        assert "Question:" in prompt, "Prompt should contain 'Question:'"
        assert question in prompt, "Prompt should contain the question"
        print("✓ format_prompt works correctly")
        
        # 测试parse_model_output
        test_cases = [
            ("Therefore, the answer is: 42", "42"),
            ("The result is 100. Therefore, the answer is: 100", "100"),
            ("#### 25", "25"),
            ("Let me calculate: 10 + 10 = 20", "20"),
        ]
        
        for output, expected in test_cases:
            result = parse_model_output(output)
            assert expected in result, f"Failed to parse '{output}', got '{result}'"
        
        print("✓ parse_model_output works correctly")
        
        # 测试答案提取
        answer_text = "Let's solve step by step:\n10 + 5 = 15\n#### 15"
        extracted = MathDataset.extract_answer(answer_text)
        assert "15" in extracted, f"Failed to extract answer from '{answer_text}'"
        print("✓ extract_answer works correctly")
        
        # 测试答案检查
        assert MathDataset.check_answer("42", "42") == True
        assert MathDataset.check_answer("42", "43") == False
        assert MathDataset.check_answer("100", "100.0") == True
        print("✓ check_answer works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """测试数据集加载（需要网络）"""
    print("\nTesting dataset loading (requires internet)...")
    
    try:
        from math_dataset import MathDataset
        
        # 尝试加载一个很小的数据集
        dataset = MathDataset(split="train[:5]", dataset_name="gsm8k")
        
        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
        
        # 测试获取样本
        sample = dataset[0]
        assert "question" in sample
        assert "final_answer" in sample
        assert "full_answer" in sample
        
        print("✓ Dataset loading works correctly")
        print(f"  Sample question: {sample['question'][:50]}...")
        print(f"  Sample answer: {sample['final_answer']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        print("  (This might be due to network issues or missing dataset)")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Component Testing Suite")
    print("="*60)
    
    results = []
    
    # 基础测试
    results.append(("Imports", test_imports()))
    
    if results[0][1]:  # 只有导入成功才继续
        results.append(("Configuration", test_config()))
        results.append(("Dataset Functions", test_dataset_functions()))
        
        # 可选测试（需要网络）
        print("\n" + "="*60)
        print("Optional Tests (require internet)")
        print("="*60)
        results.append(("Dataset Loading", test_dataset_loading()))
    
    # 打印总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
