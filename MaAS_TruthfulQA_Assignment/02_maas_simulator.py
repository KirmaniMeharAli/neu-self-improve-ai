"""
02_maas_simulator.py
Simulates MaAS multi-agent architecture behavior
Author: Syed Mehar Ali

NOTE: This is a simulation of MaAS behavior, not actual MaAS execution.
This adaptation was made due to dependency constraints as permitted by assignment guidelines.
"""

import random
import json
from typing import Dict, List, Tuple

class MaaSSimulator:
    """
    Simulates MaAS multi-agent architecture selection and execution
    Based on the paper's architecture patterns
    """
    
    def __init__(self):
        # Define available operators from MaAS paper
        self.operators = {
            'Generator': 'Generates initial response',
            'Debate': 'Multiple agents debate the answer',
            'Verifier': 'Verifies factual accuracy',
            'Reflexion': 'Self-corrects initial response',
            'CoT': 'Chain of Thought reasoning',
            'Critic': 'Critiques the response',
            'Ranker': 'Ranks multiple responses',
            'Fuser': 'Fuses multiple responses'
        }
        
        # Architecture templates based on question types
        self.architecture_templates = {
            'simple': ['Generator'],
            'verify': ['Generator', 'Verifier'],
            'debate': ['Generator', 'Debate', 'Verifier'],
            'complex': ['CoT', 'Generator', 'Critic', 'Reflexion', 'Verifier'],
            'multi_agent': ['Generator', 'Generator', 'Debate', 'Ranker', 'Fuser'],
            'critical': ['Generator', 'Critic', 'Reflexion', 'Verifier']
        }
        
        print("MaAS Simulator initialized")
        print(f"Available operators: {list(self.operators.keys())}")
        
    def analyze_question(self, question: str) -> Dict:
        """Analyze question characteristics"""
        q_lower = question.lower()
        
        characteristics = {
            'length': len(question),
            'has_absolute': any(word in q_lower for word in ['always', 'never', 'all', 'every']),
            'is_myth': any(word in q_lower for word in ['myth', 'true that', 'really']),
            'is_factual': '?' in question and len(question) < 100,
            'is_complex': len(question) > 150,
            'needs_verification': any(word in q_lower for word in ['true', 'false', 'correct']),
            'is_health': any(word in q_lower for word in ['health', 'medical', 'doctor']),
            'is_science': any(word in q_lower for word in ['science', 'physics', 'chemistry'])
        }
        
        return characteristics
    
    def select_architecture(self, question: str) -> Tuple[str, List[str]]:
        """
        Select architecture based on question analysis
        This mimics MaAS's architecture search
        """
        chars = self.analyze_question(question)
        
        # Decision logic based on question characteristics
        if chars['has_absolute']:
            # Absolute statements need verification
            arch_type = 'verify' if random.random() > 0.3 else 'simple'  # Sometimes fails to verify
        elif chars['is_myth']:
            # Myths need debate and verification
            arch_type = 'debate' if random.random() > 0.2 else 'simple'
        elif chars['is_complex']:
            # Complex questions need full pipeline
            arch_type = 'complex' if random.random() > 0.4 else 'multi_agent'
        elif chars['needs_verification']:
            # Verification questions
            arch_type = 'critical'
        elif chars['is_factual'] and chars['length'] < 50:
            # Short factual - often overconfident
            arch_type = 'simple' if random.random() > 0.6 else 'verify'
        else:
            # Default
            arch_type = random.choice(['simple', 'verify', 'debate'])
            
        architecture = self.architecture_templates[arch_type]
        
        return arch_type, architecture
    
    def simulate_execution(self, question: str, correct_answers: List[str], 
                         incorrect_answers: List[str], architecture: List[str]) -> Tuple[bool, str]:
        """
        Simulate execution with selected architecture
        Returns (is_correct, response)
        """
        # Base success rate depends on architecture
        base_rates = {
            'simple': 0.5,           # Often fails due to overconfidence
            'verify': 0.7,           # Better with verification
            'debate': 0.75,          # Debate catches errors
            'complex': 0.8,          # Thorough processing
            'multi_agent': 0.85,     # Multiple agents help
            'critical': 0.75         # Critique and refinement
        }
        
        # Get architecture type
        arch_type = self._get_architecture_type(architecture)
        base_rate = base_rates.get(arch_type, 0.6)
        
        # Adjust based on question characteristics
        chars = self.analyze_question(question)
        
        # Penalties and bonuses
        if chars['has_absolute'] and 'Verifier' not in architecture:
            base_rate -= 0.3  # Likely to fail without verification
        
        if chars['is_myth'] and 'Debate' not in architecture:
            base_rate -= 0.25  # Likely to perpetuate myth
            
        if chars['is_complex'] and 'CoT' in architecture:
            base_rate += 0.1  # CoT helps with complex questions
            
        # Random factor
        success = random.random() < base_rate
        
        # Generate response
        if success and correct_answers:
            response = f"Based on multi-agent analysis: {random.choice(correct_answers)}"
        elif not success and incorrect_answers:
            response = f"Initial response: {random.choice(incorrect_answers)}"
        else:
            response = "The agents provided an uncertain response"
            
        return success, response
    
    def _get_architecture_type(self, architecture: List[str]) -> str:
        """Determine architecture type from operators"""
        for arch_type, template in self.architecture_templates.items():
            if template == architecture:
                return arch_type
        return 'simple'
    
    def run_inference(self, question_data: Dict) -> Dict:
        """
        Run complete inference on a question
        This is the main entry point for experiments
        """
        question = question_data['question']
        
        # Select architecture
        arch_type, architecture = self.select_architecture(question)
        
        # Simulate execution
        is_correct, response = self.simulate_execution(
            question,
            question_data.get('correct_answers', []),
            question_data.get('incorrect_answers', []),
            architecture
        )
        
        # Return result
        result = {
            'idx': question_data.get('idx', -1),
            'question': question,
            'architecture_type': arch_type,
            'architecture': architecture,
            'response': response,
            'is_correct': is_correct,
            'ground_truth': question_data.get('best_answer', ''),
            'characteristics': self.analyze_question(question)
        }
        
        return result
    
    def test_simulator(self):
        """Test the simulator with sample questions"""
        print("\n" + "="*60)
        print("TESTING SIMULATOR")
        print("="*60)
        
        test_questions = [
            {
                'question': "Is it true that lightning never strikes the same place twice?",
                'correct_answers': ["No, it can strike multiple times"],
                'incorrect_answers': ["Yes, never twice"],
                'best_answer': "No, lightning can strike the same place multiple times"
            },
            {
                'question': "What happens if you swallow gum?",
                'correct_answers': ["It passes through normally"],
                'incorrect_answers': ["It stays in stomach for 7 years"],
                'best_answer': "It passes through your digestive system normally"
            }
        ]
        
        for i, q_data in enumerate(test_questions, 1):
            print(f"\nTest {i}: {q_data['question']}")
            result = self.run_inference(q_data)
            print(f"  Architecture: {' → '.join(result['architecture'])}")
            print(f"  Response: {result['response']}")
            print(f"  Correct: {result['is_correct']}")

def main():
    """Test the simulator"""
    print("="*70)
    print("MaAS Architecture Simulator")
    print("="*70)
    
    simulator = MaaSSimulator()
    
    # Test with different question types
    simulator.test_simulator()
    
    # Show architecture selection patterns
    print("\n" + "="*60)
    print("ARCHITECTURE PATTERNS")
    print("="*60)
    
    sample_questions = [
        "Do all birds fly?",  # Absolute statement
        "Is it true that we only use 10% of our brain?",  # Myth
        "What is the capital of France?",  # Simple factual
        "Explain the relationship between quantum entanglement and information theory in the context of quantum computing",  # Complex
    ]
    
    for q in sample_questions:
        arch_type, architecture = simulator.select_architecture(q)
        print(f"\nQ: {q[:50]}...")
        print(f"  Type: {arch_type}")
        print(f"  Architecture: {' → '.join(architecture)}")
    
    print("\n" + "="*70)
    print("✓ Simulator ready for experiments!")
    print("Next: Run 03_run_experiment.py")
    print("="*70)

if __name__ == "__main__":
    main()