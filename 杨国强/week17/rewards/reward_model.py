"""Reward model for evaluating math problem solutions.

This module implements a multi-component reward system that provides
dense reward signals for learning mathematical reasoning.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class RewardComponents(Enum):
    """Components of the reward signal."""
    ACCURACY = "accuracy"
    FORMAT = "format"
    STEP_PENALTY = "step_penalty"
    LENGTH_PENALTY = "length_penalty"
    PARTIAL_CREDIT = "partial_credit"


@dataclass
class RewardConfig:
    """Configuration for the reward model."""
    accuracy_weight: float = 1.0
    format_weight: float = 0.1
    step_penalty_weight: float = 0.0
    length_penalty_weight: float = -0.01
    partial_credit_weight: float = 0.5
    
    # Format requirements
    require_step_by_step: bool = True
    require_final_answer: bool = True
    max_length: int = 2000
    min_length: int = 20
    
    # Partial credit settings
    allow_partial: bool = True
    numeric_tolerance: float = 1e-6


class RewardModel:
    """Multi-component reward model for math problems.
    
    Computes rewards based on:
    1. Accuracy: Whether the final answer is correct
    2. Format: Whether the solution follows expected format
    3. Step-by-step: Whether intermediate steps are shown
    4. Length: Penalty for overly verbose/short responses
    5. Partial credit: Credit for partially correct answers
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
    def compute_reward(self, 
                      response: str,
                      problem: Any,
                      check_answer_fn=None) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and component breakdown.
        
        Args:
            response: The model's response
            problem: The problem being solved (MathProblem object)
            check_answer_fn: Optional function to check answer accuracy
            
        Returns:
            Tuple of (total_reward, component_rewards)
        """
        components = {}
        
        # 1. Accuracy reward
        accuracy_reward, is_correct = self._compute_accuracy_reward(
            response, problem, check_answer_fn
        )
        components['accuracy'] = accuracy_reward
        
        # 2. Format reward
        format_reward, format_details = self._compute_format_reward(response)
        components['format'] = format_reward
        components.update(format_details)
        
        # 3. Step penalty (for missing steps)
        step_penalty = self._compute_step_penalty(response)
        components['step_penalty'] = step_penalty
        
        # 4. Length penalty
        length_penalty = self._compute_length_penalty(response)
        components['length_penalty'] = length_penalty
        
        # 5. Partial credit
        if self.config.allow_partial and not is_correct:
            partial = self._compute_partial_credit(response, problem)
            components['partial'] = partial * self.config.partial_credit_weight
        else:
            components['partial'] = 0.0
        
        # Combine components
        total = (
            components['accuracy'] * self.config.accuracy_weight +
            components['format'] * self.config.format_weight +
            components['step_penalty'] * self.config.step_penalty_weight +
            components['length_penalty'] * self.config.length_penalty_weight +
            components['partial']
        )
        
        return total, components
    
    def _compute_accuracy_reward(self, 
                                  response: str,
                                  problem: Any,
                                  check_answer_fn) -> Tuple[float, bool]:
        """Compute accuracy reward.
        
        Returns:
            Tuple of (reward, is_correct)
        """
        if check_answer_fn is not None:
            is_correct, confidence = check_answer_fn(response, problem)
            if is_correct:
                return 1.0, True
        
        # Fallback: extract and compare answer
        extracted = self._extract_final_answer(response)
        expected = getattr(problem, 'answer', None)
        
        if extracted is None or expected is None:
            return 0.0, False
        
        # Compare
        is_correct = self._compare_answers(extracted, expected)
        
        return 1.0 if is_correct else 0.0, is_correct
    
    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from a response.
        
        Looks for patterns like:
        - "Answer: X"
        - "The answer is X"
        - "Therefore, X"
        - Final boxed answer like \\boxed{X}
        """
        # Pattern 1: Explicit answer markers
        patterns = [
            r'(?:final\s+)?(?:answer|result)\s*[:=]\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|hence)\s*,?\s*(?:the\s+)?(?:answer|result)\s+is\s+(.+?)(?:\.|$)',
            r'\\boxed\{([^}]+)\}',
            r'(?:so|solution)\s*[:=]\s*(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Pattern 2: Last number or expression at end of response
        # Look for numbers or expressions in the last few lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            # Try to find boxed answer
            boxed = re.search(r'\\boxed\{([^}]+)\}', line)
            if boxed:
                return boxed.group(1).strip()
            
            # Try to find number
            num_match = re.search(r'-?\d+\.?\d*', line)
            if num_match:
                return num_match.group(0)
        
        return None
    
    def _compare_answers(self, extracted: str, expected: Any) -> bool:
        """Compare extracted answer with expected answer."""
        # Try numeric comparison
        try:
            if isinstance(expected, (int, float)):
                ext_num = float(extracted.replace(',', ''))
                return abs(ext_num - float(expected)) < self.config.numeric_tolerance
        except (ValueError, TypeError):
            pass
        
        # String comparison
        extracted_clean = extracted.lower().strip().rstrip('.')
        expected_str = str(expected).lower().strip().rstrip('.')
        
        if extracted_clean == expected_str:
            return True
        
        # Check if extracted contains expected
        if expected_str in extracted_clean:
            return True
        
        # For lists/tuples (like quadratic solutions)
        if isinstance(expected, (list, tuple)):
            return self._compare_list_answers(extracted, expected)
        
        return False
    
    def _compare_list_answers(self, extracted: str, expected: list) -> bool:
        """Compare list answers (e.g., [2, 3] for quadratic)."""
        # Extract numbers from response
        numbers = re.findall(r'-?\d+\.?\d*', extracted)
        
        if len(numbers) < len(expected):
            return False
        
        # Check if expected numbers are present
        expected_nums = [float(x) for x in expected]
        ext_nums = [float(x) for x in numbers[:len(expected)]]
        
        return all(abs(e - actual) < self.config.numeric_tolerance 
                   for e, actual in zip(expected_nums, ext_nums))
    
    def _compute_format_reward(self, response: str) -> Tuple[float, Dict[str, float]]:
        """Compute format reward based on response structure."""
        details = {}
        
        score = 0.0
        total_checks = 0
        
        # Check 1: Has step-by-step solution
        if self.config.require_step_by_step:
            total_checks += 1
            has_steps = (
                response.count('\n') >= 2 or  # Multiple lines
                'step' in response.lower() or
                'first' in response.lower() or
                'then' in response.lower() or
                'finally' in response.lower()
            )
            details['has_steps'] = 1.0 if has_steps else 0.0
            score += details['has_steps']
        else:
            details['has_steps'] = 1.0
        
        # Check 2: Has final answer
        if self.config.require_final_answer:
            total_checks += 1
            has_answer = (
                'answer' in response.lower() or
                'therefore' in response.lower() or
                '\\boxed' in response or
                'result' in response.lower()
            )
            details['has_answer'] = 1.0 if has_answer else 0.0
            score += details['has_answer']
        else:
            details['has_answer'] = 1.0
        
        # Check 3: Proper mathematical notation
        total_checks += 1
        has_math = (
            any(op in response for op in ['=', '+', '-', '×', '÷', '*', '/', '^']) or
            '$' in response or  # LaTeX math mode
            '∫' in response or '∑' in response  # Calculus symbols
        )
        details['has_math'] = 1.0 if has_math else 0.5
        score += details['has_math']
        
        # Check 4: Coherent length
        total_checks += 1
        words = len(response.split())
        good_length = self.config.min_length <= words <= self.config.max_length
        details['good_length'] = 1.0 if good_length else 0.5
        score += details['good_length']
        
        # Normalize
        avg_score = score / total_checks if total_checks > 0 else 0.0
        
        return avg_score, details
    
    def _compute_step_penalty(self, response: str) -> float:
        """Compute penalty for missing intermediate steps."""
        if not self.config.require_step_by_step:
            return 0.0
        
        # Check if showing work
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        
        # Penalize if only final answer without steps
        if len(lines) <= 2:
            return -0.5
        
        # Check for calculation indicators
        has_calculations = any(
            c in response for c in ['=', '+', '-', '×', '÷', '*', '/', 'sqrt', '√']
        )
        
        if not has_calculations:
            return -0.3
        
        return 0.0
    
    def _compute_length_penalty(self, response: str) -> float:
        """Compute penalty for inappropriate response length."""
        words = len(response.split())
        
        if words > self.config.max_length:
            # Slight penalty for too long
            excess = words - self.config.max_length
            return max(-0.5, -excess * 0.001)
        
        if words < self.config.min_length:
            # Slight penalty for too short
            deficit = self.config.min_length - words
            return max(-0.3, -deficit * 0.005)
        
        return 0.0
    
    def _compute_partial_credit(self, response: str, problem: Any) -> float:
        """Compute partial credit for partially correct solutions."""
        credit = 0.0
        
        # 1. Check if using correct method/approach
        # (This would require more sophisticated analysis)
        
        # 2. Check if intermediate steps are correct
        expected = getattr(problem, 'answer', None)
        if expected is None:
            return 0.0
        
        # 3. For numeric answers, check magnitude
        try:
            if isinstance(expected, (int, float)):
                extracted = self._extract_final_answer(response)
                if extracted:
                    ext_num = float(extracted.replace(',', ''))
                    expected_num = float(expected)
                    
                    # Within order of magnitude
                    if abs(ext_num) > 0 and abs(expected_num) > 0:
                        ratio = min(ext_num, expected_num) / max(ext_num, expected_num)
                        if ratio > 0.9:
                            credit += 0.7
                        elif ratio > 0.5:
                            credit += 0.3
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        
        # 4. Check for correct operations even if wrong answer
        expected_ops = self._get_expected_operations(problem)
        found_ops = self._get_operations_in_response(response)
        
        if expected_ops and found_ops:
            overlap = len(expected_ops & found_ops) / len(expected_ops)
            credit += overlap * 0.2
        
        return min(credit, 0.8)  # Cap partial credit
    
    def _get_expected_operations(self, problem: Any) -> set:
        """Get expected operations for a problem."""
        ops = set()
        
        category = getattr(problem, 'category', None)
        answer = getattr(problem, 'answer', None)
        
        if category:
            cat_str = str(category).lower()
            if 'algebra' in cat_str:
                ops.add('solve')
                if isinstance(answer, (list, tuple)):
                    ops.add('quadratic')
            elif 'calculus' in cat_str:
                if 'derivative' in getattr(problem, 'prompt', '').lower():
                    ops.add('derivative')
                else:
                    ops.add('integral')
            elif 'geometry' in cat_str:
                ops.add('calculate')
                if 'area' in getattr(problem, 'prompt', '').lower():
                    ops.add('area')
                elif 'volume' in getattr(problem, 'prompt', '').lower():
                    ops.add('volume')
        
        return ops
    
    def _get_operations_in_response(self, response: str) -> set:
        """Detect operations used in the response."""
        ops = set()
        
        response_lower = response.lower()
        
        # Detection keywords
        if 'solve' in response_lower or 'solution' in response_lower:
            ops.add('solve')
        if 'derivative' in response_lower or 'differentiate' in response_lower:
            ops.add('derivative')
        if 'integrate' in response_lower or 'integral' in response_lower:
            ops.add('integral')
        if 'area' in response_lower:
            ops.add('area')
        if 'volume' in response_lower:
            ops.add('volume')
        if any(op in response for op in ['+', '-', '×', '÷', '*', '/']):
            ops.add('calculate')
        
        return ops
    
    def batch_compute(self,
                      responses: List[str],
                      problems: List[Any],
                      check_answer_fn=None) -> Tuple[List[float], List[Dict[str, float]]]:
        """Compute rewards for a batch of responses.
        
        Args:
            responses: List of model responses
            problems: List of problems
            check_answer_fn: Optional function to check answers
            
        Returns:
            Tuple of (rewards list, component dicts list)
        """
        rewards = []
        all_components = []
        
        for response, problem in zip(responses, problems):
            reward, components = self.compute_reward(
                response, problem, check_answer_fn
            )
            rewards.append(reward)
            all_components.append(components)
        
        return rewards, all_components
    
    def get_statistics(self, rewards: List[float], components: List[Dict]) -> Dict[str, float]:
        """Compute statistics over a set of rewards.
        
        Returns:
            Dictionary of statistics
        """
        if not rewards:
            return {}
        
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'accuracy_rate': np.mean([c['accuracy'] for c in components]),
            'format_score': np.mean([c['format'] for c in components]),
        }
        
        # Per-component stats
        for key in ['step_penalty', 'length_penalty', 'partial']:
            if key in components[0]:
                stats[f'mean_{key}'] = np.mean([c.get(key, 0) for c in components])
        
        return stats
