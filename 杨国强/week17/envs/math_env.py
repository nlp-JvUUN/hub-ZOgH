"""Math problem environment for GRPO training.

This module provides a comprehensive set of math problems across various
categories and difficulty levels for training language models.
"""

import random
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict


class ProblemCategory(Enum):
    """Categories of math problems."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    WORD_PROBLEMS = "word_problems"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"


class ProblemDifficulty(Enum):
    """Difficulty levels for problems."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class MathProblem:
    """Represents a math problem with its solution."""
    prompt: str
    answer: Any
    category: ProblemCategory
    difficulty: ProblemDifficulty
    solution_steps: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MathEnvironment:
    """Environment for generating and managing math problems.
    
    Provides a diverse set of math problems across categories and
    difficulty levels for training and evaluation.
    """
    
    def __init__(self, 
                 categories: Optional[List[ProblemCategory]] = None,
                 difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM,
                 seed: Optional[int] = None):
        """Initialize the math environment.
        
        Args:
            categories: List of problem categories to include
            difficulty: Default difficulty level
            seed: Random seed for reproducibility
        """
        self.categories = categories or list(ProblemCategory)
        self.difficulty = difficulty
        self.problem_bank = self._build_problem_bank()
        
        if seed is not None:
            random.seed(seed)
            
    def _build_problem_bank(self) -> Dict[Tuple[ProblemCategory, ProblemDifficulty], List[MathProblem]]:
        """Build the internal problem bank."""
        bank = defaultdict(list)
        
        # Arithmetic problems
        bank[(ProblemCategory.ARITHMETIC, ProblemDifficulty.EASY)] = self._arithmetic_easy()
        bank[(ProblemCategory.ARITHMETIC, ProblemDifficulty.MEDIUM)] = self._arithmetic_medium()
        bank[(ProblemCategory.ARITHMETIC, ProblemDifficulty.HARD)] = self._arithmetic_hard()
        
        # Algebra problems
        bank[(ProblemCategory.ALGEBRA, ProblemDifficulty.EASY)] = self._algebra_easy()
        bank[(ProblemCategory.ALGEBRA, ProblemDifficulty.MEDIUM)] = self._algebra_medium()
        bank[(ProblemCategory.ALGEBRA, ProblemDifficulty.HARD)] = self._algebra_hard()
        
        # Geometry problems
        bank[(ProblemCategory.GEOMETRY, ProblemDifficulty.EASY)] = self._geometry_easy()
        bank[(ProblemCategory.GEOMETRY, ProblemDifficulty.MEDIUM)] = self._geometry_medium()
        bank[(ProblemCategory.GEOMETRY, ProblemDifficulty.HARD)] = self._geometry_hard()
        
        # Calculus problems
        bank[(ProblemCategory.CALCULUS, ProblemDifficulty.MEDIUM)] = self._calculus_medium()
        bank[(ProblemCategory.CALCULUS, ProblemDifficulty.HARD)] = self._calculus_hard()
        
        # Word problems
        bank[(ProblemCategory.WORD_PROBLEMS, ProblemDifficulty.EASY)] = self._word_problems_easy()
        bank[(ProblemCategory.WORD_PROBLEMS, ProblemDifficulty.MEDIUM)] = self._word_problems_medium()
        
        # Number theory
        bank[(ProblemCategory.NUMBER_THEORY, ProblemDifficulty.MEDIUM)] = self._number_theory_medium()
        bank[(ProblemCategory.NUMBER_THEORY, ProblemDifficulty.HARD)] = self._number_theory_hard()
        
        # Probability
        bank[(ProblemCategory.PROBABILITY, ProblemDifficulty.MEDIUM)] = self._probability_medium()
        bank[(ProblemCategory.PROBABILITY, ProblemDifficulty.HARD)] = self._probability_hard()
        
        # Combinatorics
        bank[(ProblemCategory.COMBINATORICS, ProblemDifficulty.MEDIUM)] = self._combinatorics_medium()
        bank[(ProblemCategory.COMBINATORICS, ProblemDifficulty.HARD)] = self._combinatorics_hard()
        
        return bank
    
    def sample_problems(self, n: int, category: Optional[ProblemCategory] = None) -> List[MathProblem]:
        """Sample n problems from the problem bank.
        
        Args:
            n: Number of problems to sample
            category: Optional specific category (random from self.categories if None)
            
        Returns:
            List of sampled problems
        """
        if category is None:
            category = random.choice(self.categories)
            
        difficulty = self.difficulty
        
        available = self.problem_bank.get((category, difficulty), [])
        if not available:
            # Try other difficulties
            for diff in ProblemDifficulty:
                available = self.problem_bank.get((category, diff), [])
                if available:
                    break
        
        if not available:
            return self.sample_problems(n, None)
        
        return random.choices(available, k=n)
    
    def generate_prompt(self, problem: MathProblem, include_answer: bool = False) -> str:
        """Generate a formatted prompt for a problem.
        
        Args:
            problem: The math problem
            include_answer: Whether to include the answer in the prompt
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Solve the following {problem.category.value} problem:\n\n"
        prompt += problem.prompt
        
        if include_answer:
            prompt += f"\n\nAnswer: {problem.answer}"
            
        return prompt
    
    def check_answer(self, response: str, problem: MathProblem) -> Tuple[bool, float]:
        """Check if a response correctly solves the problem.
        
        Args:
            response: The model's response
            problem: The math problem
            
        Returns:
            Tuple of (is_correct, confidence_score)
        """
        # Try to extract answer from response
        extracted = self._extract_answer(response, problem)
        
        if extracted is None:
            return False, 0.0
            
        # Check if extracted answer matches
        is_correct = self._compare_answers(extracted, problem.answer)
        
        # Calculate confidence based on format and completeness
        confidence = 0.5 if is_correct else 0.0
        
        return is_correct, confidence
    
    def _extract_answer(self, response: str, problem: MathProblem) -> Optional[Any]:
        """Extract the final answer from a response."""
        # Try multiple extraction methods
        
        # Method 1: Look for "Answer:" or "The answer is" patterns
        answer_patterns = [
            r'(?:answer|result|solution)\s*[:=]\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|so)\s+(?:the\s+)?(?:answer|result)\s+is\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response.lower())
            if match:
                answer_str = match.group(1).strip()
                return self._parse_answer(answer_str, problem)
        
        # Method 2: Extract last number in response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            try:
                return float(numbers[-1]) if '.' in numbers[-1] else int(numbers[-1])
            except:
                pass
                
        # Method 3: For symbolic answers, check for exact match patterns
        if isinstance(problem.answer, str):
            if problem.answer.lower() in response.lower():
                return problem.answer
                
        return None
    
    def _parse_answer(self, answer_str: str, problem: MathProblem) -> Any:
        """Parse an answer string to the appropriate type."""
        answer_str = answer_str.strip()
        
        # Try to parse as number
        try:
            if '.' in answer_str:
                return float(answer_str)
            else:
                return int(answer_str)
        except ValueError:
            pass
            
        # Return as string for symbolic answers
        return answer_str
    
    def _compare_answers(self, extracted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
        """Compare extracted answer with expected answer."""
        if type(extracted) != type(expected):
            # Try numeric comparison
            try:
                extracted = float(extracted)
                expected = float(expected)
            except (ValueError, TypeError):
                return str(extracted).lower().strip() == str(expected).lower().strip()
        
        if isinstance(expected, (int, float)):
            return abs(float(extracted) - float(expected)) < tolerance
        else:
            return str(extracted).lower().strip() == str(expected).lower().strip()
    
    # ==================== Problem Generators ====================
    
    def _arithmetic_easy(self) -> List[MathProblem]:
        """Generate easy arithmetic problems."""
        problems = []
        
        # Addition/Subtraction
        for _ in range(20):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-'])
            
            if op == '+':
                ans = a + b
                prompt = f"What is {a} + {b}?"
            else:
                if b > a:
                    a, b = b, a
                ans = a - b
                prompt = f"What is {a} - {b}?"
                
            problems.append(MathProblem(
                prompt=prompt,
                answer=ans,
                category=ProblemCategory.ARITHMETIC,
                difficulty=ProblemDifficulty.EASY
            ))
        
        # Multiplication/Division
        for _ in range(20):
            a = random.randint(2, 12)
            b = random.randint(2, 12)
            op = random.choice(['*', '/'])
            
            if op == '*':
                ans = a * b
                prompt = f"What is {a} × {b}?"
            else:
                ans = a
                a = a * b
                prompt = f"What is {a} ÷ {b}?"
                
            problems.append(MathProblem(
                prompt=prompt,
                answer=ans,
                category=ProblemCategory.ARITHMETIC,
                difficulty=ProblemDifficulty.EASY
            ))
            
        return problems
    
    def _arithmetic_medium(self) -> List[MathProblem]:
        """Generate medium arithmetic problems."""
        problems = []
        
        for _ in range(30):
            ops = ['+', '-', '*', '/']
            a = random.randint(10, 100)
            b = random.randint(2, 20)
            c = random.randint(2, 10)
            
            # Random expression
            expr_type = random.randint(1, 4)
            
            if expr_type == 1:
                ans = a + b * c
                prompt = f"Calculate: {a} + {b} × {c}"
            elif expr_type == 2:
                d = random.randint(2, 10)
                ans = a * b + c * d
                prompt = f"Calculate: {a} × {b} + {c} × {d}"
            elif expr_type == 3:
                ans = (a + b) * c
                prompt = f"Calculate: ({a} + {b}) × {c}"
            else:
                ans = a * b + c
                prompt = f"Calculate: {a} × {b} + {c}"
                
            problems.append(MathProblem(
                prompt=prompt,
                answer=ans,
                category=ProblemCategory.ARITHMETIC,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _arithmetic_hard(self) -> List[MathProblem]:
        """Generate hard arithmetic problems."""
        problems = []
        
        for _ in range(20):
            a = random.randint(100, 500)
            b = random.randint(50, 200)
            c = random.randint(10, 50)
            d = random.randint(2, 10)
            
            ans = (a + b) * c // d
            prompt = f"Calculate (round to nearest integer): ({a} + {b}) × {c} ÷ {d}"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=round(ans),
                category=ProblemCategory.ARITHMETIC,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _algebra_easy(self) -> List[MathProblem]:
        """Generate easy algebra problems."""
        problems = []
        
        for _ in range(25):
            x = random.randint(1, 20)
            a = random.randint(2, 10)
            b = random.randint(1, 50)
            
            # Linear equation: ax + b = c
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=x,
                category=ProblemCategory.ALGEBRA,
                difficulty=ProblemDifficulty.EASY
            ))
            
        return problems
    
    def _algebra_medium(self) -> List[MathProblem]:
        """Generate medium algebra problems."""
        problems = []
        
        for _ in range(25):
            # Quadratic: x^2 + bx + c = 0
            r1 = random.randint(-10, 10)
            r2 = random.randint(-10, 10)
            b = -(r1 + r2)
            c = r1 * r2
            
            prompt = f"Find all solutions to: x² + {b}x + {c} = 0"
            answer = sorted([r1, r2])
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.ALGEBRA,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _algebra_hard(self) -> List[MathProblem]:
        """Generate hard algebra problems."""
        problems = []
        
        for _ in range(15):
            # System of equations
            a1, b1 = random.randint(1, 5), random.randint(1, 5)
            a2, b2 = random.randint(1, 5), random.randint(1, 5)
            x, y = random.randint(-10, 10), random.randint(-10, 10)
            
            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y
            
            prompt = f"Solve the system of equations:\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=(x, y),
                category=ProblemCategory.ALGEBRA,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _geometry_easy(self) -> List[MathProblem]:
        """Generate easy geometry problems."""
        problems = []
        
        for _ in range(15):
            # Rectangle area
            w = random.randint(2, 20)
            h = random.randint(2, 20)
            prompt = f"A rectangle has width {w} and height {h}. What is its area?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=w * h,
                category=ProblemCategory.GEOMETRY,
                difficulty=ProblemDifficulty.EASY
            ))
            
        for _ in range(15):
            # Circle area
            r = random.randint(2, 10)
            prompt = f"A circle has radius {r}. What is its area (use π ≈ 3.14159)?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=round(3.14159 * r * r, 2),
                category=ProblemCategory.GEOMETRY,
                difficulty=ProblemDifficulty.EASY
            ))
            
        return problems
    
    def _geometry_medium(self) -> List[MathProblem]:
        """Generate medium geometry problems."""
        problems = []
        
        for _ in range(20):
            # Pythagorean theorem
            a = random.randint(3, 10)
            b = random.randint(4, 10)
            c = int((a**2 + b**2)**0.5)
            
            prompt = f"In a right triangle, the two legs are {a} and {b}. What is the hypotenuse?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=c,
                category=ProblemCategory.GEOMETRY,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _geometry_hard(self) -> List[MathProblem]:
        """Generate hard geometry problems."""
        problems = []
        
        for _ in range(10):
            # Sphere volume
            r = random.randint(2, 10)
            prompt = f"A sphere has radius {r}. What is its volume (use π ≈ 3.14159)?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=round((4/3) * 3.14159 * r**3, 2),
                category=ProblemCategory.GEOMETRY,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _calculus_medium(self) -> List[MathProblem]:
        """Generate medium calculus problems."""
        problems = []
        
        for _ in range(15):
            # Basic derivatives
            n = random.randint(2, 5)
            a = random.randint(2, 5)
            
            prompt = f"What is the derivative of {a}x^{n}?"
            answer = f"{a*n}x^{n-1}"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.CALCULUS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        for _ in range(15):
            # Basic integrals
            n = random.randint(2, 5)
            prompt = f"What is the integral of x^{n} dx?"
            answer = f"x^{n+1}/({n+1}) + C"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.CALCULUS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _calculus_hard(self) -> List[MathProblem]:
        """Generate hard calculus problems."""
        problems = []
        
        for _ in range(15):
            # Product rule
            prompt = f"What is the derivative of x² sin(x)?"
            answer = "2x·sin(x) + x²·cos(x)"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.CALCULUS,
                difficulty=ProblemDifficulty.HARD
            ))
            
        for _ in range(15):
            # Chain rule
            prompt = f"What is the derivative of sin(x²)?"
            answer = "2x·cos(x²)"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.CALCULUS,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _word_problems_easy(self) -> List[MathProblem]:
        """Generate easy word problems."""
        problems = []
        
        for _ in range(20):
            apples = random.randint(5, 20)
            give = random.randint(1, apples - 1)
            prompt = f"John has {apples} apples. He gives {give} to Mary. How many does he have left?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=apples - give,
                category=ProblemCategory.WORD_PROBLEMS,
                difficulty=ProblemDifficulty.EASY
            ))
            
        for _ in range(15):
            # Rate problems
            speed = random.randint(30, 60)
            time = random.randint(2, 5)
            prompt = f"A car travels at {speed} mph for {time} hours. How far does it travel?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=speed * time,
                category=ProblemCategory.WORD_PROBLEMS,
                difficulty=ProblemDifficulty.EASY
            ))
            
        return problems
    
    def _word_problems_medium(self) -> List[MathProblem]:
        """Generate medium word problems."""
        problems = []
        
        for _ in range(20):
            # Age problems
            age1 = random.randint(20, 40)
            age2 = random.randint(5, 15)
            years = random.randint(5, 15)
            
            prompt = f"Alice is {age1} years old and her son is {age2} years old. In {years} years, how much older will Alice be than her son?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=age1 - age2,  # Age difference is constant
                category=ProblemCategory.WORD_PROBLEMS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        for _ in range(15):
            # Work problems
            time1 = random.randint(2, 6)
            time2 = random.randint(3, 8)
            
            combined = (time1 * time2) / (time1 + time2)
            
            prompt = f"Worker A can complete a task in {time1} hours and worker B in {time2} hours. How long together?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=round(combined, 2),
                category=ProblemCategory.WORD_PROBLEMS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _number_theory_medium(self) -> List[MathProblem]:
        """Generate medium number theory problems."""
        problems = []
        
        for _ in range(20):
            # GCD problems
            a = random.randint(10, 50) * 6
            b = random.randint(10, 50) * 8
            from math import gcd
            g = gcd(a, b)
            
            prompt = f"What is the GCD of {a} and {b}?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=g,
                category=ProblemCategory.NUMBER_THEORY,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        for _ in range(15):
            # Prime checking
            n = random.randint(20, 50)
            is_prime = all(n % i != 0 for i in range(2, int(n**0.5) + 1)) if n > 1 else False
            
            prompt = f"Is {n} a prime number? Answer yes or no."
            
            problems.append(MathProblem(
                prompt=prompt,
                answer="yes" if is_prime else "no",
                category=ProblemCategory.NUMBER_THEORY,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _number_theory_hard(self) -> List[MathProblem]:
        """Generate hard number theory problems."""
        problems = []
        
        for _ in range(15):
            # LCM problems
            a = random.randint(2, 10)
            b = random.randint(2, 10)
            from math import gcd
            lcm = a * b // gcd(a, b)
            
            prompt = f"What is the LCM of {a} and {b}?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=lcm,
                category=ProblemCategory.NUMBER_THEORY,
                difficulty=ProblemDifficulty.HARD
            ))
            
        for _ in range(15):
            # Modular arithmetic
            a = random.randint(10, 50)
            b = random.randint(2, 10)
            c = a % b
            
            prompt = f"What is {a} mod {b}?"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=c,
                category=ProblemCategory.NUMBER_THEORY,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _probability_medium(self) -> List[MathProblem]:
        """Generate medium probability problems."""
        problems = []
        
        for _ in range(20):
            # Coin flip
            n = random.randint(2, 5)
            prompt = f"What is the probability of getting exactly {n} heads when flipping {n*2} fair coins?"
            
            from math import comb
            total = 2 ** (n * 2)
            favorable = comb(n * 2, n)
            answer = round(favorable / total, 4)
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.PROBABILITY,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        for _ in range(15):
            # Dice problems
            n = random.randint(1, 3)
            prompt = f"What is the probability of rolling a sum of {n*3 + 3} with two dice?"
            
            # Count favorable outcomes
            favorable = sum(1 for i in range(1, 7) for j in range(1, 7) if i + j == n*3 + 3)
            answer = f"{favorable}/36"
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.PROBABILITY,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _probability_hard(self) -> List[MathProblem]:
        """Generate hard probability problems."""
        problems = []
        
        for _ in range(15):
            # Conditional probability
            prompt = f"A bag has 5 red and 3 blue balls. Two balls are drawn without replacement. What is P(both red)?"
            answer = round((5/8) * (4/7), 4)
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.PROBABILITY,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
    
    def _combinatorics_medium(self) -> List[MathProblem]:
        """Generate medium combinatorics problems."""
        problems = []
        
        from math import comb, factorial
        
        for _ in range(20):
            n = random.randint(5, 10)
            r = random.randint(2, 5)
            prompt = f"How many ways can we choose {r} items from {n} items?"
            answer = comb(n, r)
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.COMBINATORICS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        for _ in range(15):
            n = random.randint(4, 8)
            prompt = f"How many permutations of the word '{'ABCDEFGH'[:n]}' are possible?"
            answer = factorial(n)
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.COMBINATORICS,
                difficulty=ProblemDifficulty.MEDIUM
            ))
            
        return problems
    
    def _combinatorics_hard(self) -> List[MathProblem]:
        """Generate hard combinatorics problems."""
        problems = []
        
        from math import comb
        
        for _ in range(20):
            # Binomial expansion
            prompt = f"What is the coefficient of x³ in (x + 2)⁶?"
            answer = comb(6, 3) * (2 ** 3)
            
            problems.append(MathProblem(
                prompt=prompt,
                answer=answer,
                category=ProblemCategory.COMBINATORICS,
                difficulty=ProblemDifficulty.HARD
            ))
            
        return problems
