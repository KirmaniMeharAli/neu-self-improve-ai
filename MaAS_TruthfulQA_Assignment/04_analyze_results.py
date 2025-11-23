"""
04_analyze_results.py
Deep analysis of experiment results for assignment
Author: Syed Mehar Ali
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ResultsAnalyzer:
    def __init__(self, results_dir=None):
        """Initialize analyzer"""
        if results_dir:
            self.results_dir = results_dir
        else:
            # Load from last experiment
            if os.path.exists('last_experiment.txt'):
                with open('last_experiment.txt', 'r') as f:
                    self.results_dir = f.read().strip()
            else:
                # Find most recent
                results_folders = [d for d in os.listdir('results') if d.startswith('experiment_')]
                if results_folders:
                    self.results_dir = f"results/{sorted(results_folders)[-1]}"
                else:
                    raise ValueError("No results found! Run experiment first.")
        
        print(f"Loading results from: {self.results_dir}")
        
        # Load results
        with open(f"{self.results_dir}/results.json", 'r') as f:
            self.results = json.load(f)
        
        # Separate failures and successes
        self.failures = [r for r in self.results if not r['is_correct']]
        self.successes = [r for r in self.results if r['is_correct']]
        
        print(f"Loaded {len(self.results)} results")
        print(f"  - {len(self.successes)} successes")
        print(f"  - {len(self.failures)} failures")
    
    def analyze_failures(self, n=5):
        """
        Analyze the n easiest failures in detail
        This is required for the assignment
        """
        print("\n" + "="*70)
        print(f"ANALYZING {n} EASIEST FAILURES")
        print("="*70)
        
        # Sort by question length (easiest = shortest)
        sorted_failures = sorted(self.failures, key=lambda x: len(x['question']))[:n]
        
        failure_analysis = []
        
        for i, failure in enumerate(sorted_failures, 1):
            print(f"\n{'='*60}")
            print(f"FAILURE {i}/{n}")
            print('='*60)
            
            analysis = self.deep_analyze_failure(failure)
            failure_analysis.append(analysis)
            
            # Print detailed analysis
            print(f"Question: {failure['question']}")
            print(f"Expected: {failure['ground_truth']}")
            print(f"MaAS Response: {failure['response']}")
            print(f"Architecture Used: {' → '.join(failure['architecture'])}")
            
            print(f"\n📊 ROOT CAUSE ANALYSIS:")
            print(f"  Primary Cause: {analysis['root_cause']}")
            print(f"  Question Type: {analysis['question_type']}")
            
            print(f"\n🔍 MISSING OPERATORS:")
            if analysis['missing_operators']:
                for op in analysis['missing_operators']:
                    print(f"  ✗ {op}")
            else:
                print("  None identified")
            
            print(f"\n⚠️ SEARCH FAILURE:")
            print(f"  {analysis['search_failure_reason']}")
        
        # Save analysis
        with open(f"{self.results_dir}/failure_analysis.json", 'w') as f:
            json.dump(failure_analysis, f, indent=2)
        
        print(f"\n✓ Saved detailed failure analysis")
        return failure_analysis
    
    def deep_analyze_failure(self, failure):
        """Detailed analysis of a single failure"""
        question = failure['question'].lower()
        architecture = failure['architecture']
        chars = failure.get('characteristics', {})
        
        analysis = {
            'idx': failure['idx'],
            'question': failure['question'],
            'ground_truth': failure['ground_truth'],
            'maas_response': failure['response'],
            'architecture': architecture,
            'architecture_type': failure['architecture_type'],
            'root_cause': '',
            'question_type': '',
            'missing_operators': [],
            'search_failure': False,
            'search_failure_reason': ''
        }
        
        # Determine question type and root cause
        if chars.get('has_absolute'):
            analysis['question_type'] = 'Absolute Statement'
            if 'Verifier' not in architecture:
                analysis['root_cause'] = 'Failed to verify absolute claim'
                analysis['missing_operators'].append('Verifier - Would check factual accuracy')
                analysis['search_failure'] = True
                analysis['search_failure_reason'] = 'Search selected simple architecture for absolute statement'
            else:
                analysis['root_cause'] = 'Verification present but failed'
        
        elif chars.get('is_myth'):
            analysis['question_type'] = 'Common Misconception'
            if 'Debate' not in architecture:
                analysis['root_cause'] = 'No debate to challenge misconception'
                analysis['missing_operators'].append('Debate - Would challenge the myth')
            if 'Verifier' not in architecture:
                analysis['missing_operators'].append('Verifier - Would fact-check')
            if not analysis['root_cause']:
                analysis['root_cause'] = 'Architecture inadequate for myth-busting'
        
        elif len(failure['question']) < 50:
            analysis['question_type'] = 'Short Factual'
            analysis['root_cause'] = 'Overconfidence on simple-seeming question'
            if failure['architecture_type'] == 'simple':
                analysis['missing_operators'].append('Verifier - Even simple questions need verification')
                analysis['search_failure'] = True
                analysis['search_failure_reason'] = 'Incorrectly classified as too simple to need verification'
        
        else:
            analysis['question_type'] = 'General Knowledge'
            analysis['root_cause'] = 'Incorrect information in base response'
            if 'Reflexion' not in architecture:
                analysis['missing_operators'].append('Reflexion - Would allow self-correction')
        
        # Check for search failure patterns
        if failure['architecture_type'] == 'simple' and (chars.get('has_absolute') or chars.get('is_myth')):
            analysis['search_failure'] = True
            analysis['search_failure_reason'] = 'Architecture search failed to identify question complexity'
        
        if not analysis['search_failure_reason']:
            analysis['search_failure_reason'] = 'No search failure detected - architecture was reasonable but execution failed'
        
        return analysis
    
    def analyze_successes(self, n=5):
        """
        Analyze the n hardest successes
        This is required for the assignment
        """
        print("\n" + "="*70)
        print(f"ANALYZING {n} HARDEST SUCCESSES")
        print("="*70)
        
        # Sort by question length (hardest = longest)
        sorted_successes = sorted(self.successes, key=lambda x: len(x['question']), reverse=True)[:n]
        
        success_analysis = []
        
        for i, success in enumerate(sorted_successes, 1):
            print(f"\n{'='*60}")
            print(f"SUCCESS {i}/{n}")
            print('='*60)
            
            analysis = self.deep_analyze_success(success)
            success_analysis.append(analysis)
            
            # Print analysis
            print(f"Question: {success['question'][:100]}...")
            print(f"Correct Answer: {success['ground_truth']}")
            print(f"MaAS Response: {success['response']}")
            
            print(f"\n🏗️ ARCHITECTURE USED:")
            self.print_architecture_diagram(success['architecture'])
            
            print(f"\n✅ WHY IT WORKED:")
            for reason in analysis['why_it_worked']:
                print(f"  • {reason}")
        
        # Save analysis
        with open(f"{self.results_dir}/success_analysis.json", 'w') as f:
            json.dump(success_analysis, f, indent=2)
        
        print(f"\n✓ Saved detailed success analysis")
        return success_analysis
    
    def deep_analyze_success(self, success):
        """Detailed analysis of why a success worked"""
        architecture = success['architecture']
        chars = success.get('characteristics', {})
        
        analysis = {
            'idx': success['idx'],
            'question': success['question'],
            'correct_answer': success['ground_truth'],
            'maas_response': success['response'],
            'architecture': architecture,
            'architecture_type': success['architecture_type'],
            'why_it_worked': []
        }
        
        # Analyze why each operator helped
        if 'Debate' in architecture:
            analysis['why_it_worked'].append(
                "Debate operator: Multiple agents challenged each other's assumptions"
            )
        
        if 'Verifier' in architecture:
            analysis['why_it_worked'].append(
                "Verifier operator: Ensured factual accuracy before final answer"
            )
        
        if 'CoT' in architecture:
            analysis['why_it_worked'].append(
                "Chain-of-Thought: Systematic step-by-step reasoning through complexity"
            )
        
        if 'Reflexion' in architecture:
            analysis['why_it_worked'].append(
                "Reflexion operator: Self-corrected initial response"
            )
        
        if 'Critic' in architecture:
            analysis['why_it_worked'].append(
                "Critic operator: Identified weaknesses in initial response"
            )
        
        if 'Ranker' in architecture and 'Fuser' in architecture:
            analysis['why_it_worked'].append(
                "Ranker + Fuser: Selected and combined best aspects from multiple responses"
            )
        
        # Architecture-level analysis
        if success['architecture_type'] == 'complex':
            analysis['why_it_worked'].append(
                "Complex architecture: Comprehensive pipeline handled nuanced question"
            )
        elif success['architecture_type'] == 'multi_agent':
            analysis['why_it_worked'].append(
                "Multi-agent collaboration: Diverse perspectives led to correct answer"
            )
        
        return analysis
    
    def print_architecture_diagram(self, architecture):
        """Print a visual representation of the architecture"""
        print("\n    [Input]")
        print("       ↓")
        for i, operator in enumerate(architecture):
            print(f"    [{operator}]")
            if i < len(architecture) - 1:
                print("       ↓")
        print("    [Output]")
    
    def create_visualizations(self):
        """Create visualizations for the report"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # 1. Architecture usage pie chart
        arch_usage = {}
        for r in self.results:
            arch_type = r['architecture_type']
            arch_usage[arch_type] = arch_usage.get(arch_type, 0) + 1
        
        plt.figure(figsize=(10, 6))
        plt.pie(arch_usage.values(), labels=arch_usage.keys(), autopct='%1.1f%%')
        plt.title('Architecture Type Distribution')
        plt.savefig(f'{self.results_dir}/architecture_distribution.png', dpi=100)
        plt.close()
        
        # 2. Success rate by architecture
        arch_success = {}
        arch_total = {}
        for r in self.results:
            arch_type = r['architecture_type']
            arch_total[arch_type] = arch_total.get(arch_type, 0) + 1
            if r['is_correct']:
                arch_success[arch_type] = arch_success.get(arch_type, 0) + 1
        
        success_rates = {k: (arch_success.get(k, 0) / v * 100) for k, v in arch_total.items()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(success_rates.keys(), success_rates.values())
        plt.xlabel('Architecture Type')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Architecture Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/success_by_architecture.png', dpi=100)
        plt.close()
        
        print("✓ Created visualizations")

def main():
    """Main execution"""
    print("="*70)
    print("DEEP ANALYSIS FOR ASSIGNMENT")
    print("Syed Mehar Ali - INFO 6105")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer()
    
    # Analyze failures (Assignment requirement 1)
    failure_analysis = analyzer.analyze_failures(n=5)
    
    # Analyze successes (Assignment requirement 2)
    success_analysis = analyzer.analyze_successes(n=5)
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {analyzer.results_dir}/failure_analysis.json")
    print(f"  - {analyzer.results_dir}/success_analysis.json")
    print(f"  - {analyzer.results_dir}/architecture_distribution.png")
    print(f"  - {analyzer.results_dir}/success_by_architecture.png")
    print("\nNext: Run 05_generate_report.py for final submission")

if __name__ == "__main__":
    main()