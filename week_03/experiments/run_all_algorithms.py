"""
Run all control algorithms on Windy Gridworld
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')

from windy_gridworld import WindyGridworld
from algorithms.dynamic_programming import DynamicProgramming
from algorithms.monte_carlo import MonteCarloControl
from algorithms.temporal_difference import TemporalDifference


def run_all_algorithms():
    """
    Run all control algorithms and compare their performance
    """
    print("=" * 80)
    print("WINDY GRIDWORLD - CONTROL ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Create environment
    env = WindyGridworld(height=7, width=10, stochastic=True)
    
    # Store results
    results = {}
    
    # 1. Dynamic Programming Control
    print("\n" + "=" * 80)
    print("1. DYNAMIC PROGRAMMING CONTROL")
    print("=" * 80)
    
    dp = DynamicProgramming(env, gamma=0.9)
    
    # Policy Iteration
    start_time = time.time()
    V_pi, policy_pi = dp.policy_iteration(theta=1e-4)
    pi_time = time.time() - start_time
    
    print(f"\nPolicy Iteration completed in {pi_time:.2f} seconds")
    env.render(value_function=V_pi, policy=policy_pi)
    
    results['DP Control (Policy Iteration)'] = {
        'V': V_pi,
        'policy': policy_pi,
        'time': pi_time
    }
    
    # 2. Monte Carlo Control
    print("\n" + "=" * 80)
    print("2. MONTE CARLO CONTROL")
    print("=" * 80)
    
    # MC On-policy
    mc = MonteCarloControl(env, gamma=0.9)
    start_time = time.time()
    Q_mc_on, episodes_mc_on = mc.on_policy_mc_control(n_episodes=5000, epsilon=0.1)
    mc_on_time = time.time() - start_time
    
    print(f"\nMC On-Policy Control completed in {mc_on_time:.2f} seconds")
    
    results['MC On-Policy Control'] = {
        'Q': Q_mc_on,
        'episodes': episodes_mc_on,
        'time': mc_on_time
    }
    
    # MC Off-policy
    mc_off = MonteCarloControl(env, gamma=0.9)
    start_time = time.time()
    Q_mc_off, episodes_mc_off = mc_off.off_policy_mc_control(n_episodes=5000, behavior_epsilon=0.1)
    mc_off_time = time.time() - start_time
    
    print(f"\nMC Off-Policy Control completed in {mc_off_time:.2f} seconds")
    
    results['MC Off-Policy Control'] = {
        'Q': Q_mc_off,
        'episodes': episodes_mc_off,
        'time': mc_off_time
    }
    
    # 3. Temporal Difference Control
    print("\n" + "=" * 80)
    print("3. TEMPORAL DIFFERENCE CONTROL")
    print("=" * 80)
    
    # TD(0) On-Policy Control: SARSA
    td_sarsa = TemporalDifference(env, gamma=0.9)
    start_time = time.time()
    Q_sarsa, episodes_sarsa = td_sarsa.sarsa(n_episodes=500, alpha=0.5, epsilon=0.1)
    sarsa_time = time.time() - start_time
    
    print(f"\nTD(0) On-Policy Control completed in {sarsa_time:.2f} seconds")
    
    results['TD(0) On-Policy'] = {
        'Q': Q_sarsa,
        'episodes': episodes_sarsa,
        'time': sarsa_time
    }
    
    # TD(0) Off-Policy with Unweighted Importance Sampling
    td_unweighted = TemporalDifference(env, gamma=0.9)
    start_time = time.time()
    Q_unweighted, episodes_unweighted = td_unweighted.td_off_policy_unweighted_is(
        n_episodes=500, alpha=0.5, behavior_epsilon=0.1
    )
    unweighted_time = time.time() - start_time
    
    print(f"\nTD(0) Off-Policy Unweighted IS completed in {unweighted_time:.2f} seconds")
    
    results['TD(0) Off-Policy Unweighted IS'] = {
        'Q': Q_unweighted,
        'episodes': episodes_unweighted,
        'time': unweighted_time
    }
    
    # TD(0) Off-Policy with Weighted Importance Sampling
    td_weighted = TemporalDifference(env, gamma=0.9)
    start_time = time.time()
    Q_weighted, episodes_weighted = td_weighted.td_off_policy_weighted_is(
        n_episodes=500, alpha=0.5, behavior_epsilon=0.1
    )
    weighted_time = time.time() - start_time
    
    print(f"\nTD(0) Off-Policy Weighted IS completed in {weighted_time:.2f} seconds")
    
    results['TD(0) Off-Policy Weighted IS'] = {
        'Q': Q_weighted,
        'episodes': episodes_weighted,
        'time': weighted_time
    }
    
    # Create comparison plots
    create_comparison_plots(results, env)
    
    return results


def create_comparison_plots(results, env):
    """
    Create comparison plots for all algorithms
    
    Args:
        results: Dictionary of algorithm results
        env: WindyGridworld environment
    """
    # Plot 1: Episode lengths over time (for episodic algorithms)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title('Learning Curves - Episode Lengths')
    
    episodic_algorithms = ['MC On-Policy Control', 'MC Off-Policy Control',
                          'TD(0) On-Policy', 'TD(0) Off-Policy Unweighted IS', 
                          'TD(0) Off-Policy Weighted IS']
    
    for algo in episodic_algorithms:
        if algo in results and 'episodes' in results[algo]:
            episodes = results[algo]['episodes']
            # Smooth with moving average
            window = 50
            if len(episodes) > window:
                smoothed = np.convolve(episodes, np.ones(window)/window, mode='valid')
                plt.plot(smoothed, label=algo, alpha=0.7)
    
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Computation time comparison
    plt.subplot(2, 2, 2)
    plt.title('Computation Time Comparison')
    
    algorithms = list(results.keys())
    times = [results[algo]['time'] for algo in algorithms]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    bars = plt.bar(range(len(algorithms)), times, color=colors)
    plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Final path comparison
    plt.subplot(2, 2, 3)
    plt.title('Optimal Paths Comparison')
    
    # Test each policy
    path_lengths = {}
    
    for algo in algorithms:
        if 'Q' in results[algo]:
            Q = results[algo]['Q']
            # Extract greedy policy
            policy = np.zeros((env.height * env.width, env.n_actions))
            for s in range(env.height * env.width):
                policy[s, np.argmax(Q[s])] = 1.0
        elif 'policy' in results[algo]:
            policy = results[algo]['policy']
        else:
            continue
            
        # Simulate path
        pos = env.reset()
        path_length = 0
        max_steps = 100
        
        for _ in range(max_steps):
            state_idx = env.position_to_state(pos)
            action = np.argmax(policy[state_idx])
            pos, _, done = env.step(action)
            path_length += 1
            
            if done:
                break
                
        path_lengths[algo] = path_length if done else max_steps
    
    # Plot path lengths
    algo_names = list(path_lengths.keys())
    lengths = list(path_lengths.values())
    
    bars = plt.bar(range(len(algo_names)), lengths, color=colors[:len(algo_names)])
    plt.xticks(range(len(algo_names)), algo_names, rotation=45, ha='right')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{length}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Algorithm summary table
    plt.subplot(2, 2, 4)
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Algorithm', 'Type', 'Time (s)', 'Path']
    
    for algo in algorithms[:6]:  # Limit to 6 for display
        if 'DP' in algo:
            algo_type = 'DP'
        elif 'MC' in algo:
            algo_type = 'MC'
        else:
            algo_type = 'TD'
        time_val = f"{results[algo]['time']:.2f}"
        path_len = str(path_lengths.get(algo, 'N/A'))
        
        # Shorten algorithm names for table
        short_name = algo.replace('Control', '').replace('Policy Iteration', 'PI')
        table_data.append([short_name, algo_type, time_val, path_len])
    
    table = plt.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Windy Gridworld - Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison plots saved as 'algorithm_comparison.png'")


if __name__ == "__main__":
    # Run all algorithms
    results = run_all_algorithms()
    
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    print("\nAlgorithm Performance Summary:")
    print("-" * 60)
    
    for algo, data in results.items():
        print(f"\n{algo}:")
        print(f"  Computation time: {data['time']:.3f} seconds")
        
        if 'episodes' in data:
            final_episodes = data['episodes'][-100:] if len(data['episodes']) > 100 else data['episodes']
            avg_steps = np.mean(final_episodes)
            print(f"  Average steps (last 100 episodes): {avg_steps:.1f}")
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)