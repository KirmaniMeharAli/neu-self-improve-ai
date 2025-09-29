"""
Helper utilities for RL algorithms
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import List, Tuple, Dict
import random


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


def epsilon_greedy_action(Q_values: np.ndarray, epsilon: float) -> int:
    """
    Select action using epsilon-greedy strategy
    
    Args:
        Q_values: Q-values for current state
        epsilon: Exploration probability
        
    Returns:
        Selected action
    """
    if random.random() < epsilon:
        return random.randint(0, len(Q_values) - 1)
    else:
        # Break ties randomly
        max_q = np.max(Q_values)
        best_actions = np.where(Q_values == max_q)[0]
        return random.choice(best_actions)


def calculate_returns(episode: List[Tuple[int, int, float]], 
                     gamma: float = 0.9) -> List[float]:
    """
    Calculate returns for each step in an episode
    
    Args:
        episode: List of (state, action, reward) tuples
        gamma: Discount factor
        
    Returns:
        List of returns
    """
    returns = []
    G = 0
    
    for _, _, reward in reversed(episode):
        G = gamma * G + reward
        returns.insert(0, G)
        
    return returns


def moving_average(data: List[float], window: int) -> np.ndarray:
    """
    Calculate moving average
    
    Args:
        data: Input data
        window: Window size
        
    Returns:
        Smoothed data
    """
    if len(data) < window:
        return np.array(data)
        
    return np.convolve(data, np.ones(window)/window, mode='valid')


def evaluate_greedy_policy(env, Q: np.ndarray, 
                          n_episodes: int = 100,
                          max_steps: int = 1000) -> Dict[str, float]:
    """
    Evaluate a greedy policy derived from Q-function
    
    Args:
        env: Environment
        Q: Q-function
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    episode_lengths = []
    successes = 0
    
    for _ in range(n_episodes):
        pos = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            state_idx = env.position_to_state(pos)
            action = np.argmax(Q[state_idx])
            
            next_pos, reward, done = env.step(action)
            total_reward += reward
            
            if done:
                successes += 1
                episode_lengths.append(step + 1)
                break
                
            pos = next_pos
        else:
            episode_lengths.append(max_steps)
            
        total_rewards.append(total_reward)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': successes / n_episodes
    }


def print_algorithm_summary(name: str, results: Dict[str, float]):
    """
    Print formatted algorithm results
    
    Args:
        name: Algorithm name
        results: Results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"{name} - Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Mean Reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:     {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate:    {results['success_rate']*100:.1f}%")


def save_results_to_file(results: Dict, filename: str = "results.txt"):
    """
    Save experiment results to file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("Self-Improving AI - Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        
        for algo, data in results.items():
            f.write(f"{algo}:\n")
            f.write("-" * 40 + "\n")
            
            for key, value in data.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                elif isinstance(value, int):
                    f.write(f"  {key}: {value}\n")
                    
            f.write("\n")
    
    print(f"Results saved to {filename}")


def create_state_visitation_heatmap(env, policy, n_episodes: int = 100):
    """
    Create heatmap of state visitations under a policy
    
    Args:
        env: Environment
        policy: Policy to evaluate
        n_episodes: Number of episodes to run
        
    Returns:
        Visitation counts array
    """
    visitations = np.zeros(env.height * env.width)
    
    for _ in range(n_episodes):
        pos = env.reset()
        
        for _ in range(1000):  # Max steps
            state_idx = env.position_to_state(pos)
            visitations[state_idx] += 1
            
            if policy.ndim == 2:
                action = np.argmax(policy[state_idx])
            else:
                action = policy[state_idx]
                
            next_pos, _, done = env.step(action)
            
            if done:
                break
                
            pos = next_pos
            
    return visitations.reshape(env.height, env.width)