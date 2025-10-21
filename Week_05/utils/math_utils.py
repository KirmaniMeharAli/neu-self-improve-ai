"""
Mathematical utilities for MATH dataset
"""
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Optional, Union, List, Tuple
import numpy as np

class MathExpressionParser:
    """Parser for mathematical expressions and solutions."""
    
    def __init__(self):
        """Initialize parser with sympy transformations."""
        self.transformations = (standard_transformations + (implicit_multiplication_application,))
    
    def parse_expression(self, expr_str: str) -> Optional[sp.Expr]:
        """
        Parse mathematical expression string.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Sympy expression or None if parsing fails
        """
        try:
            # Clean expression
            expr_str = self.clean_expression(expr_str)
            
            # Parse with sympy
            expr = parse_expr(
                expr_str,
                transformations=self.transformations
            )
            return expr
        except:
            return None
    
    def clean_expression(self, expr_str: str) -> str:
        """Clean expression string for parsing."""
        # Remove extra spaces
        expr_str = ' '.join(expr_str.split())
        
        # Replace common LaTeX commands
        replacements = {
            '\\cdot': '*',
            '\\times': '*',
            '\\div': '/',
            '\\frac': 'frac',
            '\\sqrt': 'sqrt',
            '\\pi': 'pi',
            '\\infty': 'oo',
            '\\sum': 'Sum',
            '\\prod': 'Product',
            '\\int': 'Integral',
            '\\sin': 'sin',
            '\\cos': 'cos',
            '\\tan': 'tan',
            '\\log': 'log',
            '\\ln': 'ln',
            '\\exp': 'exp'
        }
        
        for latex, sympy in replacements.items():
            expr_str = expr_str.replace(latex, sympy)
        
        # Handle fractions
        expr_str = self.convert_fractions(expr_str)
        
        return expr_str
    
    def convert_fractions(self, expr_str: str) -> str:
        """Convert LaTeX fractions to sympy format."""
        # Pattern for \frac{numerator}{denominator}
        frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
        
        def replace_frac(match):
            num, den = match.groups()
            return f"({num})/({den})"
        
        return re.sub(frac_pattern, replace_frac, expr_str)
    
    def extract_numeric_value(self, expr: Union[str, sp.Expr]) -> Optional[float]:
        """
        Extract numeric value from expression.
        
        Args:
            expr: Expression (string or sympy)
            
        Returns:
            Numeric value or None
        """
        try:
            if isinstance(expr, str):
                expr = self.parse_expression(expr)
            
            if expr is None:
                return None
            
            # Evaluate to float
            value = float(expr.evalf())
            return value
        except:
            return None
    
    def compare_expressions(self, expr1: str, expr2: str, tolerance: float = 1e-9) -> bool:
        """
        Compare two mathematical expressions.
        
        Args:
            expr1: First expression
            expr2: Second expression
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if expressions are equivalent
        """
        try:
            # Parse expressions
            parsed1 = self.parse_expression(expr1)
            parsed2 = self.parse_expression(expr2)
            
            if parsed1 is None or parsed2 is None:
                # Fall back to string comparison
                return self.normalize_string(expr1) == self.normalize_string(expr2)
            
            # Try symbolic comparison
            diff = sp.simplify(parsed1 - parsed2)
            
            # Check if difference is zero
            if diff == 0:
                return True
            
            # Try numerical comparison
            try:
                diff_val = abs(float(diff.evalf()))
                return diff_val < tolerance
            except:
                return False
                
        except:
            # Fall back to string comparison
            return self.normalize_string(expr1) == self.normalize_string(expr2)
    
    def normalize_string(self, s: str) -> str:
        """Normalize string for comparison."""
        # Remove whitespace
        s = ''.join(s.split())
        # Convert to lowercase
        s = s.lower()
        # Remove common punctuation
        s = s.replace(',', '').replace('$', '')
        return s

class MathSolutionVerifier:
    """Verifier for mathematical solutions."""
    
    def __init__(self):
        """Initialize verifier."""
        self.parser = MathExpressionParser()
        self.answer_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'The answer is[:\s]+([^\n.]+)',
            r'Therefore[,:\s]+([^\n.]+)',
            r'Final answer[:\s]+([^\n.]+)',
            r'Solution[:\s]+([^\n.]+)',
            r'=\s*([^=\n]+)$'
        ]
    
    def extract_answer(self, solution: str) -> Optional[str]:
        """
        Extract final answer from solution.
        
        Args:
            solution: Solution text
            
        Returns:
            Extracted answer or None
        """
        # Try each pattern
        for pattern in self.answer_patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Try to find last mathematical expression
        math_exprs = re.findall(r'[0-9+\-*/=().\s]+', solution)
        if math_exprs:
            # Get last substantial expression
            for expr in reversed(math_exprs):
                if len(expr.strip()) > 2 and '=' in expr:
                    parts = expr.split('=')
                    if len(parts) >= 2:
                        return parts[-1].strip()
        
        # Return last line as fallback
        lines = solution.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return None
    
    def verify_solution(self, solution: str, expected_answer: str) -> Tuple[bool, str]:
        """
        Verify if solution contains correct answer.
        
        Args:
            solution: Generated solution
            expected_answer: Expected answer
            
        Returns:
            Tuple of (is_correct, extracted_answer)
        """
        # Extract answer from solution
        extracted = self.extract_answer(solution)
        
        if extracted is None:
            return False, ""
        
        # Compare answers
        is_correct = self.parser.compare_expressions(extracted, expected_answer)
        
        return is_correct, extracted
    
    def check_solution_steps(self, solution: str) -> Dict:
        """
        Check quality of solution steps.
        
        Args:
            solution: Solution text
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'has_steps': False,
            'has_equations': False,
            'has_final_answer': False,
            'num_steps': 0,
            'num_equations': 0,
            'solution_length': len(solution.split()),
            'is_well_structured': False
        }
        
        # Check for steps
        step_indicators = ['step', 'first', 'second', 'then', 'next', 'finally']
        metrics['has_steps'] = any(ind in solution.lower() for ind in step_indicators)
        
        # Count steps
        step_matches = re.findall(r'step \d+|first|second|then|next|finally', solution.lower())
        metrics['num_steps'] = len(step_matches)
        
        # Check for equations
        equation_matches = re.findall(r'[^=]=[^=]', solution)
        metrics['num_equations'] = len(equation_matches)
        metrics['has_equations'] = metrics['num_equations'] > 0
        
        # Check for final answer
        metrics['has_final_answer'] = any(
            re.search(pattern, solution, re.IGNORECASE)
            for pattern in self.answer_patterns
        )
        
        # Check structure
        metrics['is_well_structured'] = (
            metrics['has_steps'] and 
            metrics['has_equations'] and 
            metrics['has_final_answer'] and
            metrics['solution_length'] > 20
        )
        
        return metrics

class MathProblemClassifier:
    """Classifier for math problem types."""
    
    def __init__(self):
        """Initialize classifier."""
        self.type_keywords = {
            'algebra': ['solve', 'equation', 'variable', 'x', 'y', 'polynomial', 'factor'],
            'geometry': ['triangle', 'circle', 'angle', 'area', 'perimeter', 'square', 'rectangle'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate', 'integrate'],
            'probability': ['probability', 'chance', 'likelihood', 'random', 'expected'],
            'number_theory': ['prime', 'divisor', 'gcd', 'lcm', 'modulo', 'congruent'],
            'combinatorics': ['combination', 'permutation', 'arrange', 'choose', 'ways']
        }
    
    def classify_problem(self, problem_text: str) -> str:
        """
        Classify problem type.
        
        Args:
            problem_text: Problem statement
            
        Returns:
            Problem type string
        """
        problem_lower = problem_text.lower()
        
        scores = {}
        for prob_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                scores[prob_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def estimate_difficulty(self, problem_text: str) -> str:
        """
        Estimate problem difficulty.
        
        Args:
            problem_text: Problem statement
            
        Returns:
            Difficulty level (easy, medium, hard)
        """
        # Simple heuristics
        problem_length = len(problem_text.split())
        num_equations = len(re.findall(r'[^=]=[^=]', problem_text))
        has_complex_notation = bool(re.search(r'\\[a-zA-Z]+', problem_text))
        
        score = 0
        
        # Length-based scoring
        if problem_length > 100:
            score += 2
        elif problem_length > 50:
            score += 1
        
        # Equation complexity
        if num_equations > 3:
            score += 2
        elif num_equations > 1:
            score += 1
        
        # Notation complexity
        if has_complex_notation:
            score += 1
        
        # Map score to difficulty
        if score >= 4:
            return 'hard'
        elif score >= 2:
            return 'medium'
        else:
            return 'easy'

# Utility functions
def format_latex_solution(solution: str) -> str:
    """Format solution with proper LaTeX."""
    # Add proper equation delimiters
    solution = re.sub(r'(\$[^$]+\$)', r'\[\1\]', solution)
    
    # Format final answer
    if '\\boxed{' not in solution:
        # Try to extract and box the final answer
        lines = solution.strip().split('\n')
        if lines:
            last_line = lines[-1]
            if '=' in last_line:
                parts = last_line.split('=')
                if len(parts) >= 2:
                    answer = parts[-1].strip()
                    lines[-1] = f"{parts[0]} = \\boxed{{{answer}}}"
                    solution = '\n'.join(lines)
    
    return solution

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers from text."""
    numbers = []
    
    # Find all numeric patterns
    patterns = [
        r'-?\d+\.?\d*',  # Regular numbers
        r'-?\d+/\d+',     # Fractions
        r'-?\d+\.?\d*e[+-]?\d+'  # Scientific notation
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                if '/' in match:
                    # Handle fraction
                    parts = match.split('/')
                    value = float(parts[0]) / float(parts[1])
                elif 'e' in match.lower():
                    # Handle scientific notation
                    value = float(match)
                else:
                    value = float(match)
                numbers.append(value)
            except:
                continue
    
    return numbers