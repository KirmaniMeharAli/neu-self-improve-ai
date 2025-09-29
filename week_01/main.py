"""
Created on Sep 23 02:04:22 2025

@author: mehar

This is the main script that runs all experiments and generates the results.

"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import our modules
from environment import BanditEnvironment
from agents import EpsilonGreedyAgent, GradientBanditAgent
from experiment import ExperimentRunner
from visualization import Visualizer


def run_all_experiments():
    """
    Run all experiments required for the assignment.
    """
    print("="*70)
    print("MULTI-ARMED BANDIT EXPERIMENTS")
    print("Reproducing Figure 2.2 with Gradient Policy Learning")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize experiment runner
    runner = ExperimentRunner(n_runs=2000, n_steps=1000, n_arms=10)
    
    # Dictionary to store all results
    all_results = {}
    
    # ========================================
    # PART 1: Epsilon-Greedy Experiments
    # ========================================
    print("\n" + "-"*70)
    print("PART 1: EPSILON-GREEDY METHODS")
    print("-"*70)
    
    # Experiment 1: Pure Greedy (ε = 0)
    result = runner.run_experiment(
        EpsilonGreedyAgent,
        {'epsilon': 0.0},
        'ε=0 (greedy)'
    )
    all_results['ε=0 (greedy)'] = result
    
    # Experiment 2: ε = 0.01
    result = runner.run_experiment(
        EpsilonGreedyAgent,
        {'epsilon': 0.01},
        'ε=0.01'
    )
    all_results['ε=0.01'] = result
    
    # Experiment 3: ε = 0.1
    result = runner.run_experiment(
        EpsilonGreedyAgent,
        {'epsilon': 0.1},
        'ε=0.1'
    )
    all_results['ε=0.1'] = result
    
    # ========================================
    # PART 2: Gradient Bandit Experiments
    # ========================================
    print("\n" + "-"*70)
    print("PART 2: GRADIENT BANDIT METHODS")
    print("-"*70)
    
    # Experiment 4: Gradient with α = 0.1, no baseline
    result = runner.run_experiment(
        GradientBanditAgent,
        {'learning_rate': 0.1, 'use_baseline': False},
        'gradient α=0.1 (no baseline)'
    )
    all_results['gradient α=0.1 (no baseline)'] = result
    
    # Experiment 5: Gradient with α = 0.4, no baseline
    result = runner.run_experiment(
        GradientBanditAgent,
        {'learning_rate': 0.4, 'use_baseline': False},
        'gradient α=0.4 (no baseline)'
    )
    all_results['gradient α=0.4 (no baseline)'] = result
    
    # Experiment 6: Gradient with α = 0.1, with baseline
    result = runner.run_experiment(
        GradientBanditAgent,
        {'learning_rate': 0.1, 'use_baseline': True},
        'gradient α=0.1 (baseline)'
    )
    all_results['gradient α=0.1 (baseline)'] = result
    
    # Experiment 7: Gradient with α = 0.4, with baseline
    result = runner.run_experiment(
        GradientBanditAgent,
        {'learning_rate': 0.4, 'use_baseline': True},
        'gradient α=0.4 (baseline)'
    )
    all_results['gradient α=0.4 (baseline)'] = result
    
    return runner, all_results


def generate_visualizations(all_results):
    """
    Generate all required visualizations.
    """
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    # Initialize visualizer
    viz = Visualizer(style='seaborn-v0_8-darkgrid')
    
    # Create main Figure 2.2 reproduction
    fig = viz.create_figure_2_2(all_results)
    plt.show()
    
    return viz


def print_final_analysis(runner, all_results):
    """
    Print detailed analysis of results.
    """
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    
    # Print summary table
    runner.print_summary()
    
    # Find best performing method
    best_reward = -float('inf')
    best_optimal = -float('inf')
    best_reward_method = None
    best_optimal_method = None
    
    for name in all_results:
        perf = runner.get_final_performance(name)
        if perf['avg_reward'] > best_reward:
            best_reward = perf['avg_reward']
            best_reward_method = name
        if perf['avg_optimal'] > best_optimal:
            best_optimal = perf['avg_optimal']
            best_optimal_method = name
    
    print("\n" + "-"*70)
    print("BEST PERFORMERS")
    print("-"*70)
    print(f"Highest Average Reward: {best_reward_method} ({best_reward:.3f})")
    print(f"Highest % Optimal Actions: {best_optimal_method} ({best_optimal:.1f}%)")
    
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)
    print("1. Gradient methods with baseline show superior performance")
    print("2. Higher learning rates (α=0.4) converge faster but with more variance")
    print("3. Baseline reduces variance in gradient estimates significantly")
    print("4. Pure greedy (ε=0) performs poorly due to lack of exploration")
    
    print("\n" + "="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


def main():
    """
    Main execution function.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all experiments
    runner, all_results = run_all_experiments()
    
    # Generate visualizations
    viz = generate_visualizations(all_results)
    
    # Print final analysis
    print_final_analysis(runner, all_results)
    
    print("\n✅ All experiments completed successfully!")
    print("📊 Figure saved as 'figure_2_2_reproduction.png'")
    print("📝 Results have been analyzed and summarized above")


if __name__ == "__main__":
    main()
