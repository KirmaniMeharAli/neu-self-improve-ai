"""
Created on Sep 22 02:04:22 2025

@author: mehar

This module handles running experiments and collecting results.

"""

import numpy as np
from typing import Dict, List, Tuple
import time


class ExperimentRunner:
    """
    Manages and runs bandit experiments.
    """
    
    def __init__(self, n_runs=2000, n_steps=1000, n_arms=10):
        """
        Initialize experiment runner.
        
        Parameters:
        -----------
        n_runs : int
            Number of independent runs to average over
        n_steps : int
            Number of steps per run
        n_arms : int
            Number of arms in the bandit
        """
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.n_arms = n_arms
        
        # Store results
        self.results = {}
    
    def run_single_simulation(self, agent, environment):
        """
        Run a single simulation with given agent and environment.
        
        Parameters:
        -----------
        agent : BaseAgent
            The agent to evaluate
        environment : BanditEnvironment
            The bandit environment
        
        Returns:
        --------
        tuple
            (rewards, optimal_actions) arrays for each step
        """
        rewards = np.zeros(self.n_steps)
        optimal_actions = np.zeros(self.n_steps)
        
        for step in range(self.n_steps):
            # Agent selects action
            action = agent.select_action()
            
            # Environment provides reward
            reward = environment.pull_arm(action)
            
            # Agent learns from reward
            agent.update(action, reward)
            
            # Record results
            rewards[step] = reward
            optimal_actions[step] = environment.is_optimal_action(action)
        
        return rewards, optimal_actions
    
    def run_experiment(self, agent_class, agent_params, experiment_name):
        """
        Run multiple simulations and average results.
        
        Parameters:
        -----------
        agent_class : class
            Agent class to instantiate
        agent_params : dict
            Parameters for agent initialization
        experiment_name : str
            Name for this experiment
        
        Returns:
        --------
        dict
            Experiment results
        """
        print(f"\nRunning experiment: {experiment_name}")
        print(f"  Parameters: {agent_params}")
        print(f"  Running {self.n_runs} simulations...")
        
        # Arrays to store results from all runs
        all_rewards = np.zeros((self.n_runs, self.n_steps))
        all_optimal = np.zeros((self.n_runs, self.n_steps))
        
        start_time = time.time()
        
        # Progress tracking
        progress_markers = [int(self.n_runs * p) for p in [0.25, 0.5, 0.75]]
        
        for run in range(self.n_runs):
            # Progress indicator
            if run in progress_markers:
                elapsed = time.time() - start_time
                print(f"    {int(100 * run / self.n_runs)}% complete ({elapsed:.1f}s)")
            
            # Import here to avoid circular dependency
            from environment import BanditEnvironment
            
            # Create fresh environment and agent for each run
            env = BanditEnvironment(self.n_arms, random_seed=run)
            agent = agent_class(n_arms=self.n_arms, **agent_params)
            
            # Run simulation
            rewards, optimal = self.run_single_simulation(agent, env)
            
            # Store results
            all_rewards[run] = rewards
            all_optimal[run] = optimal
        
        # Calculate statistics
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        pct_optimal = np.mean(all_optimal, axis=0) * 100
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f} seconds")
        
        # Store results
        result = {
            'name': experiment_name,
            'avg_rewards': avg_rewards,
            'std_rewards': std_rewards,
            'pct_optimal': pct_optimal,
            'params': agent_params,
            'time_elapsed': elapsed
        }
        
        self.results[experiment_name] = result
        
        return result
    
    def get_final_performance(self, experiment_name, last_n_steps=100):
        """
        Get average performance over the last n steps.
        
        Parameters:
        -----------
        experiment_name : str
            Name of experiment
        last_n_steps : int
            Number of final steps to average
        
        Returns:
        --------
        dict
            Final performance metrics
        """
        if experiment_name not in self.results:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        result = self.results[experiment_name]
        
        return {
            'avg_reward': np.mean(result['avg_rewards'][-last_n_steps:]),
            'avg_optimal': np.mean(result['pct_optimal'][-last_n_steps:])
        }
    
    def print_summary(self):
        """Print summary of all experiment results."""
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        print(f"{'Experiment':<35} {'Avg Reward':>12} {'% Optimal':>12} {'Time (s)':>10}")
        print("-"*70)
        
        for name, result in self.results.items():
            perf = self.get_final_performance(name)
            print(f"{name:<35} {perf['avg_reward']:>12.3f} {perf['avg_optimal']:>11.1f}% {result['time_elapsed']:>10.1f}")
        
        print("="*70)