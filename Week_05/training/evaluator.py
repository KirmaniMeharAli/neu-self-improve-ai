"""
Evaluator for PAG with A*-PO on MATH dataset
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import re
from sympy import sympify, simplify
from sympy.parsing.sympy_parser import parse_expr
import warnings
warnings.filterwarnings('ignore')

class MATHEvaluator:
    """Evaluator for MATH dataset solutions."""
    
    def __init__(self, pag_system, tokenizer, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            pag_system: PAG system with policy and value models
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.pag = pag_system
        self.tokenizer = tokenizer
        self.device = device
        
        # Evaluation metrics
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset evaluation metrics."""
        self.metrics = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'by_level': {},
            'by_type': {},
            'solution_lengths': [],
            'mcts_values': [],
            'mcts_visits': [],
            'errors': []
        }
    
    def evaluate_dataset(self, dataloader, num_samples: Optional[int] = None) -> Dict:
        """
        Evaluate on entire dataset.
        
        Args:
            dataloader: DataLoader with evaluation data
            num_samples: Limit number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.reset_metrics()
        
        total_batches = len(dataloader)
        if num_samples:
            total_batches = min(total_batches, num_samples // dataloader.batch_size)
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=total_batches,
            desc="Evaluating"
        )
        
        with torch.no_grad():
            for batch_idx, batch in progress_bar:
                if num_samples and batch_idx >= total_batches:
                    break
                
                # Evaluate batch
                batch_results = self.evaluate_batch(batch)
                
                # Update metrics
                self._update_metrics(batch_results, batch)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'acc': f"{self.metrics['accuracy']:.3f}",
                    'correct': self.metrics['correct'],
                    'total': self.metrics['total']
                })
        
        # Compute final statistics
        self._compute_final_stats()
        
        return self.metrics
    
    def evaluate_batch(self, batch: Dict) -> List[Dict]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            List of evaluation results for each sample
        """
        results = []
        
        for i in range(len(batch['problem'])):
            # Extract sample
            problem = batch['problem'][i]
            target_answer = batch['final_answer'][i]
            level = batch['level'][i] if batch['level'] else None
            problem_type = batch['type'][i] if batch['type'] else None
            
            # Generate solution
            solution_dict = self.pag.generate_solution(problem)
            
            # Check correctness
            is_correct = self.check_answer(
                solution_dict['solution'],
                target_answer
            )
            
            # Store result
            result = {
                'problem': problem,
                'generated_solution': solution_dict['solution'],
                'target_answer': target_answer,
                'is_correct': is_correct,
                'level': level,
                'type': problem_type,
                'mcts_value': solution_dict.get('mcts_value', 0),
                'mcts_visits': solution_dict.get('mcts_visits', 0),
                'solution_length': len(solution_dict['solution'].split())
            }
            
            results.append(result)
        
        return results
    
    def check_answer(self, solution: str, target_answer: str) -> bool:
        """
        Check if generated solution contains correct answer.
        
        Args:
            solution: Generated solution text
            target_answer: Ground truth answer
            
        Returns:
            Boolean indicating if answer is correct
        """
        try:
            # Extract answer from solution
            pred_answer = self.extract_answer(solution)
            
            # Normalize answers
            pred_normalized = self.normalize_answer(pred_answer)
            target_normalized = self.normalize_answer(target_answer)
            
            # Direct string comparison
            if pred_normalized == target_normalized:
                return True
            
            # Try numeric comparison
            if self.is_numeric(pred_normalized) and self.is_numeric(target_normalized):
                pred_num = float(pred_normalized)
                target_num = float(target_normalized)
                return abs(pred_num - target_num) < 1e-6
            
            # Try symbolic comparison for algebraic expressions
            try:
                pred_expr = parse_expr(pred_normalized)
                target_expr = parse_expr(target_normalized)
                return simplify(pred_expr - target_expr) == 0
            except:
                pass
            
            return False
            
        except Exception as e:
            self.metrics['errors'].append({
                'error': str(e),
                'solution': solution[:200],
                'target': target_answer
            })
            return False
    
    def extract_answer(self, solution: str) -> str:
        """
        Extract final answer from solution.
        
        Args:
            solution: Solution text
            
        Returns:
            Extracted answer string
        """
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Look for "The answer is" pattern
        answer_match = re.search(
            r'(?:The answer is|Final answer:|Therefore,?)\s*([^\n.]+)',
            solution,
            re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()
        
        # Look for last equation
        equations = re.findall(r'=\s*([^=\n]+)(?:\n|$)', solution)
        if equations:
            return equations[-1].strip()
        
        # Extract last number
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        if numbers:
            return numbers[-1]
        
        # Return last line as fallback
        lines = solution.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.
        
        Args:
            answer: Answer string
            
        Returns:
            Normalized answer
        """
        # Remove whitespace and convert to lowercase
        answer = answer.strip().lower()
        
        # Remove dollar signs and commas
        answer = answer.replace('$', '').replace(',', '')
        
        # Remove leading/trailing punctuation
        answer = answer.strip('.')
        
        # Handle fractions
        if '/' in answer:
            try:
                parts = answer.split('/')
                if len(parts) == 2:
                    num, den = float(parts[0]), float(parts[1])
                    answer = str(num / den)
            except:
                pass
        
        # Remove units (simplified)
        units = ['cm', 'mm', 'm', 'km', 'kg', 'g', 'l', 'ml', 'degrees', 'degree']
        for unit in units:
            answer = answer.replace(unit, '').strip()
        
        return answer
    
    def is_numeric(self, s: str) -> bool:
        """Check if string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _update_metrics(self, results: List[Dict], batch: Dict):
        """Update metrics with batch results."""
        for result in results:
            self.metrics['total'] += 1
            if result['is_correct']:
                self.metrics['correct'] += 1
            
            # Update by level
            if result['level']:
                if result['level'] not in self.metrics['by_level']:
                    self.metrics['by_level'][result['level']] = {
                        'correct': 0, 'total': 0
                    }
                self.metrics['by_level'][result['level']]['total'] += 1
                if result['is_correct']:
                    self.metrics['by_level'][result['level']]['correct'] += 1
            
            # Update by type
            if result['type']:
                if result['type'] not in self.metrics['by_type']:
                    self.metrics['by_type'][result['type']] = {
                        'correct': 0, 'total': 0
                    }
                self.metrics['by_type'][result['type']]['total'] += 1
                if result['is_correct']:
                    self.metrics['by_type'][result['type']]['correct'] += 1
            
            # Collect other metrics
            self.metrics['solution_lengths'].append(result['solution_length'])
            self.metrics['mcts_values'].append(result['mcts_value'])
            self.metrics['mcts_visits'].append(result['mcts_visits'])
        
        # Update accuracy
        if self.metrics['total'] > 0:
            self.metrics['accuracy'] = self.metrics['correct'] / self.metrics['total']
    
    def _compute_final_stats(self):
        """Compute final statistics."""
        # Overall accuracy
        if self.metrics['total'] > 0:
            self.metrics['accuracy'] = self.metrics['correct'] / self.metrics['total']
        
        # Level-wise accuracy
        for level in self.metrics['by_level']:
            total = self.metrics['by_level'][level]['total']
            if total > 0:
                correct = self.metrics['by_level'][level]['correct']
                self.metrics['by_level'][level]['accuracy'] = correct / total
        
        # Type-wise accuracy
        for prob_type in self.metrics['by_type']:
            total = self.metrics['by_type'][prob_type]['total']
            if total > 0:
                correct = self.metrics['by_type'][prob_type]['correct']
                self.metrics['by_type'][prob_type]['accuracy'] = correct / total
        
        # Average metrics
        if self.metrics['solution_lengths']:
            self.metrics['avg_solution_length'] = np.mean(self.metrics['solution_lengths'])
            self.metrics['avg_mcts_value'] = np.mean(self.metrics['mcts_values'])
            self.metrics['avg_mcts_visits'] = np.mean(self.metrics['mcts_visits'])
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"Overall Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Correct: {self.metrics['correct']} / {self.metrics['total']}")
        
        if self.metrics['by_level']:
            print("\nAccuracy by Level:")
            for level in sorted(self.metrics['by_level'].keys()):
                stats = self.metrics['by_level'][level]
                acc = stats.get('accuracy', 0)
                print(f"  {level}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
        if self.metrics['by_type']:
            print("\nAccuracy by Type:")
            for prob_type in sorted(self.metrics['by_type'].keys()):
                stats = self.metrics['by_type'][prob_type]
                acc = stats.get('accuracy', 0)
                print(f"  {prob_type}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
        if 'avg_solution_length' in self.metrics:
            print(f"\nAverage Solution Length: {self.metrics['avg_solution_length']:.1f} words")
            print(f"Average MCTS Value: {self.metrics['avg_mcts_value']:.4f}")
            print(f"Average MCTS Visits: {self.metrics['avg_mcts_visits']:.1f}")
        
        if self.metrics['errors']:
            print(f"\nErrors encountered: {len(self.metrics['errors'])}")
        
        print("="*50)
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"Results saved to {filepath}")