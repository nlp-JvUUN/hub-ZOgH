"""Evaluation script for trained GRPO models."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from core.policy import Policy, create_policy, MockPolicy
from envs.math_env import MathEnvironment, ProblemCategory, ProblemDifficulty, MathProblem
from rewards.reward_model import RewardModel, RewardConfig


class MathEvaluator:
    """Evaluator for math problem solving performance."""
    
    def __init__(self, 
                 policy: Policy,
                 env: MathEnvironment,
                 reward_model: RewardModel):
        self.policy = policy
        self.env = env
        self.reward_model = reward_model
    
    def evaluate(self, 
                 n_problems: int = 100,
                 temperature: float = 0.0,
                 verbose: bool = False) -> Dict[str, Any]:
        """Evaluate on a set of problems.
        
        Args:
            n_problems: Number of problems to evaluate
            temperature: Sampling temperature (0 for greedy)
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        problems = self.env.sample_problems(n_problems)
        
        results = {
            'total': n_problems,
            'correct': 0,
            'incorrect': 0,
            'rewards': [],
            'per_category': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'per_difficulty': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'responses': []
        }
        
        for problem in tqdm(problems, desc="Evaluating"):
            # Generate response
            prompt = self.env.generate_prompt(problem)
            responses = self.policy.generate(
                [prompt],
                temperature=temperature,
                max_tokens=1024
            )
            
            response_text = responses[0]['response']
            
            # Check answer
            is_correct, _ = self.env.check_answer(response_text, problem)
            
            # Compute reward
            reward, components = self.reward_model.compute_reward(
                response_text, problem, check_answer_fn=self.env.check_answer
            )
            
            # Record results
            results['correct'] += int(is_correct)
            results['rewards'].append(reward)
            
            # Per-category stats
            cat_name = problem.category.value
            results['per_category'][cat_name]['total'] += 1
            results['per_category'][cat_name]['correct'] += int(is_correct)
            
            # Per-difficulty stats
            diff_name = problem.difficulty.value
            results['per_difficulty'][diff_name]['total'] += 1
            results['per_difficulty'][diff_name]['correct'] += int(is_correct)
            
            # Store response details if verbose
            if verbose:
                results['responses'].append({
                    'prompt': prompt,
                    'response': response_text,
                    'expected_answer': problem.answer,
                    'is_correct': is_correct,
                    'reward': reward,
                    'components': components
                })
        
        # Compute summary metrics
        results['accuracy'] = results['correct'] / n_problems
        results['mean_reward'] = np.mean(results['rewards'])
        results['std_reward'] = np.std(results['rewards'])
        results['min_reward'] = np.min(results['rewards'])
        results['max_reward'] = np.max(results['rewards'])
        
        # Per-category accuracy
        results['per_category_accuracy'] = {
            cat: data['correct'] / data['total'] 
            for cat, data in results['per_category'].items()
        }
        
        # Per-difficulty accuracy
        results['per_difficulty_accuracy'] = {
            diff: data['correct'] / data['total']
            for diff, data in results['per_difficulty'].items()
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:     {results['accuracy']:.2%}")
        print(f"  Mean Reward:  {results['mean_reward']:.4f}")
        print(f"  Std Reward:    {results['std_reward']:.4f}")
        print(f"  Correct:      {results['correct']}/{results['total']}")
        
        print(f"\nReward Statistics:")
        print(f"  Min:  {results['min_reward']:.4f}")
        print(f"  Max:  {results['max_reward']:.4f}")
        
        print(f"\nPer-Category Accuracy:")
        for cat, acc in results['per_category_accuracy'].items():
            data = results['per_category'][cat]
            print(f"  {cat:20s}: {acc:6.2%} ({data['correct']}/{data['total']})")
        
        print(f"\nPer-Difficulty Accuracy:")
        for diff, acc in results['per_difficulty_accuracy'].items():
            data = results['per_difficulty'][diff]
            print(f"  {diff:20s}: {acc:6.2%} ({data['correct']}/{data['total']})")
        
        print("\n" + "="*60)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert non-serializable types
        serializable = {
            'accuracy': results['accuracy'],
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'min_reward': results['min_reward'],
            'max_reward': results['max_reward'],
            'correct': results['correct'],
            'total': results['total'],
            'per_category': {k: dict(v) for k, v in results['per_category'].items()},
            'per_difficulty': {k: dict(v) for k, v in results['per_difficulty'].items()},
            'per_category_accuracy': results['per_category_accuracy'],
            'per_difficulty_accuracy': results['per_difficulty_accuracy'],
        }
        
        # Add responses if present
        if 'responses' in results:
            serializable['responses'] = results['responses']
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def create_evaluator(args) -> MathEvaluator:
    """Create an evaluator based on command line arguments."""
    # Create policy
    if args.policy_type == 'mock':
        policy = MockPolicy(model_name=args.model_name)
    else:
        policy = create_policy(
            args.policy_type,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    
    # Create environment
    categories = [ProblemCategory(c) for c in args.categories] if args.categories else None
    env = MathEnvironment(
        categories=categories or [ProblemCategory.ARITHMETIC, ProblemCategory.ALGEBRA],
        difficulty=ProblemDifficulty(args.difficulty),
        seed=args.seed
    )
    
    # Create reward model
    reward_config = RewardConfig(
        accuracy_weight=1.0,
        format_weight=0.1,
        length_penalty_weight=-0.01,
        partial_credit_weight=0.5
    )
    reward_model = RewardModel(reward_config)
    
    return MathEvaluator(policy, env, reward_model)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GRPO math model")
    
    # Evaluation
    parser.add_argument("--n_problems", type=int, default=100,
                        help="Number of problems to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    
    # Policy
    parser.add_argument("--policy_type", type=str, default="mock",
                        choices=["mock", "openai", "anthropic"],
                        help="Policy type")
    parser.add_argument("--model_name", type=str, default="gpt-4",
                        help="Model name")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    
    # Environment
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "expert"],
                        help="Problem difficulty")
    parser.add_argument("--categories", nargs="+",
                        default=["arithmetic", "algebra", "geometry"],
                        help="Problem categories")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--show_examples", type=int, default=0,
                        help="Show N example problems and responses")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("Creating evaluator...")
    evaluator = create_evaluator(args)
    
    print(f"Evaluating on {args.n_problems} problems...")
    results = evaluator.evaluate(
        n_problems=args.n_problems,
        temperature=args.temperature,
        verbose=args.verbose or args.show_examples > 0
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if args.output:
        evaluator.save_results(results, args.output)
    
    # Show examples
    if args.show_examples > 0 and 'responses' in results:
        print("\n" + "="*60)
        print(f"SAMPLE PROBLEMS AND RESPONSES (showing {args.show_examples})")
        print("="*60)
        
        for i, resp in enumerate(results['responses'][:args.show_examples]):
            print(f"\n--- Example {i+1} ---")
            print(f"Problem: {resp['prompt'][:200]}...")
            print(f"Response: {resp['response'][:300]}...")
            print(f"Expected: {resp['expected_answer']}")
            print(f"Correct: {resp['is_correct']}")
            print(f"Reward: {resp['reward']:.4f}")


if __name__ == "__main__":
    main()
