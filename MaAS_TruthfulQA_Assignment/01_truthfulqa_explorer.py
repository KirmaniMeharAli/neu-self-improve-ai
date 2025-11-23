"""
01_truthfulqa_explorer.py
Explores and analyzes the TruthfulQA dataset
Author: Syed Mehar Ali
"""

import json
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

class TruthfulQAExplorer:
    def __init__(self):
        print("Loading TruthfulQA dataset...")
        self.dataset = load_dataset("truthful_qa", "generation")
        self.validation = self.dataset['validation']
        print(f"✓ Loaded {len(self.validation)} questions\n")
        
    def explore_dataset(self):
        """Explore dataset characteristics"""
        print("="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Basic statistics
        questions = [item['question'] for item in self.validation]
        answers = [item['best_answer'] for item in self.validation]
        
        q_lengths = [len(q) for q in questions]
        a_lengths = [len(a) for a in answers]
        
        print(f"\nDataset Statistics:")
        print(f"  Total questions: {len(questions)}")
        print(f"  Avg question length: {sum(q_lengths)/len(q_lengths):.1f} chars")
        print(f"  Avg answer length: {sum(a_lengths)/len(a_lengths):.1f} chars")
        print(f"  Shortest question: {min(q_lengths)} chars")
        print(f"  Longest question: {max(q_lengths)} chars")
        
    def categorize_questions(self):
        """Categorize questions by type"""
        categories = {
            'misconceptions': [],
            'absolute_statements': [],
            'factual': [],
            'health_myths': [],
            'science': [],
            'history': [],
            'other': []
        }
        
        for idx, item in enumerate(self.validation):
            q_lower = item['question'].lower()
            
            if any(word in q_lower for word in ['always', 'never', 'every', 'all', 'none']):
                categories['absolute_statements'].append(idx)
            elif any(word in q_lower for word in ['myth', 'true that', 'really']):
                categories['misconceptions'].append(idx)
            elif any(word in q_lower for word in ['health', 'doctor', 'medical', 'disease']):
                categories['health_myths'].append(idx)
            elif any(word in q_lower for word in ['science', 'physics', 'chemistry']):
                categories['science'].append(idx)
            elif any(word in q_lower for word in ['history', 'historical', 'war']):
                categories['history'].append(idx)
            elif '?' in q_lower and len(q_lower) < 100:
                categories['factual'].append(idx)
            else:
                categories['other'].append(idx)
        
        print("\n" + "="*60)
        print("QUESTION CATEGORIES")
        print("="*60)
        
        for category, indices in categories.items():
            print(f"  {category:20}: {len(indices):4} questions")
            
        return categories
    
    def find_easy_and_hard(self, n=10):
        """Find potentially easy and hard questions"""
        # Sort by length as a proxy for difficulty
        sorted_by_length = sorted(
            enumerate(self.validation), 
            key=lambda x: len(x[1]['question'])
        )
        
        print("\n" + "="*60)
        print(f"TOP {n} POTENTIALLY EASY QUESTIONS (shortest)")
        print("="*60)
        
        easy_candidates = []
        for i, (idx, item) in enumerate(sorted_by_length[:n], 1):
            print(f"\n{i}. [{idx}] {item['question']}")
            print(f"   Answer: {item['best_answer'][:80]}...")
            easy_candidates.append(idx)
            
        print("\n" + "="*60)
        print(f"TOP {n} POTENTIALLY HARD QUESTIONS (longest)")
        print("="*60)
        
        hard_candidates = []
        for i, (idx, item) in enumerate(sorted_by_length[-n:], 1):
            print(f"\n{i}. [{idx}] {item['question'][:100]}...")
            print(f"   Answer: {item['best_answer'][:80]}...")
            hard_candidates.append(idx)
            
        return easy_candidates, hard_candidates
    
    def save_sample_questions(self, num_samples=100):
        """Save sample questions for experiment"""
        samples = []
        for idx in range(min(num_samples, len(self.validation))):
            item = self.validation[idx]
            samples.append({
                'idx': idx,
                'question': item['question'],
                'best_answer': item['best_answer'],
                'correct_answers': item['correct_answers'],
                'incorrect_answers': item['incorrect_answers']
            })
        
        # Save to JSON
        output_file = 'data/sample_questions.json'
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
            
        print(f"\n✓ Saved {len(samples)} sample questions to {output_file}")
        return samples
    
    def visualize_distribution(self):
        """Create visualization of question lengths"""
        q_lengths = [len(item['question']) for item in self.validation]
        
        plt.figure(figsize=(10, 6))
        plt.hist(q_lengths, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Question Length (characters)')
        plt.ylabel('Count')
        plt.title('TruthfulQA Question Length Distribution')
        plt.axvline(sum(q_lengths)/len(q_lengths), color='red', 
                   linestyle='--', label=f'Mean: {sum(q_lengths)/len(q_lengths):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('visualizations/question_distribution.png', dpi=100, bbox_inches='tight')
        print("\n✓ Saved visualization to visualizations/question_distribution.png")
        plt.show()

def main():
    print("="*70)
    print("TruthfulQA Dataset Explorer")
    print("="*70)
    
    explorer = TruthfulQAExplorer()
    
    # Explore dataset
    explorer.explore_dataset()
    
    # Categorize questions
    categories = explorer.categorize_questions()
    
    # Find easy and hard questions
    easy, hard = explorer.find_easy_and_hard(n=5)
    
    # Save sample questions
    explorer.save_sample_questions(num_samples=100)
    
    # Create visualization
    explorer.visualize_distribution()
    
    print("\n" + "="*70)
    print("✓ Dataset exploration complete!")
    print("Next: Run 02_maas_simulator.py")
    print("="*70)

if __name__ == "__main__":
    main()