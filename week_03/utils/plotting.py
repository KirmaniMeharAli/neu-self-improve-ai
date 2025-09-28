"""
Plotting utilities for visualization
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def plot_value_function(V, env, title="Value Function"):
    """
    Plot value function as heatmap
    
    Args:
        V: Value function array
        env: Environment object
        title: Plot title
    """
    # Reshape V to grid
    V_grid = V.reshape(env.height, env.width)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(V_grid, annot=True, fmt='.1f', cmap='coolwarm', 
                center=0, cbar_kws={'label': 'Value'})
    
    # Mark special positions
    plt.scatter(env.start_pos[1] + 0.5, env.start_pos[0] + 0.5, 
               marker='s', s=200, c='green', label='Start')
    plt.scatter(env.goal_pos[1] + 0.5, env.goal_pos[0] + 0.5,
               marker='*', s=300, c='gold', label='Goal')
    
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.legend(loc='best')
    plt.tight_layout()
    
    return plt.gcf()


def plot_policy(policy, env, title="Policy"):
    """
    Plot policy with arrows
    
    Args:
        policy: Policy array (states x actions) or Q-function
        env: Environment object
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grid
    for i in range(env.height + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(env.width + 1):
        ax.axvline(j, color='gray', linewidth=0.5)
    
    # Arrow directions
    arrow_map = {
        0: (0, -0.3),   # Up
        1: (0.3, 0),    # Right
        2: (0, 0.3),    # Down
        3: (-0.3, 0)    # Left
    }
    
    # Draw policy arrows
    for row in range(env.height):
        for col in range(env.width):
            state_idx = env.position_to_state((row, col))
            
            if (row, col) == env.goal_pos:
                ax.add_patch(Rectangle((col, row), 1, 1, 
                                      facecolor='gold', alpha=0.5))
                ax.text(col + 0.5, row + 0.5, 'GOAL', 
                       ha='center', va='center', fontweight='bold')
            elif (row, col) == env.start_pos:
                ax.add_patch(Rectangle((col, row), 1, 1,
                                      facecolor='lightgreen', alpha=0.5))
                ax.text(col + 0.5, row + 0.5, 'START',
                       ha='center', va='center', fontweight='bold')
            else:
                # Get best action
                if policy.ndim == 2:
                    best_action = np.argmax(policy[state_idx])
                else:
                    best_action = np.argmax(policy[state_idx])
                
                dx, dy = arrow_map[best_action]
                ax.arrow(col + 0.5, row + 0.5, dx, dy,
                        head_width=0.1, head_length=0.1,
                        fc='blue', ec='blue', alpha=0.7)
    
    # Add wind indicators
    for col in range(env.width):
        wind = env.wind_strength[col] if col < len(env.wind_strength) else 0
        if wind > 0:
            ax.text(col + 0.5, env.height + 0.3, f'↑{wind}',
                   ha='center', va='center', color='red', fontsize=12)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Column (Wind strength shown above)')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    return fig


def plot_episode_lengths(episode_lengths, title="Episode Lengths Over Time",
                         window_size=50):
    """
    Plot episode lengths with smoothing
    
    Args:
        episode_lengths: List of episode lengths
        title: Plot title
        window_size: Window for moving average
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.plot(episode_lengths, alpha=0.3, label='Raw')
    
    # Plot smoothed data
    if len(episode_lengths) > window_size:
        smoothed = np.convolve(episode_lengths, 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        plt.plot(range(window_size-1, len(episode_lengths)), 
                smoothed, label=f'Smoothed (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_learning_curves(results_dict, title="Learning Curves Comparison"):
    """
    Plot learning curves for multiple algorithms
    
    Args:
        results_dict: Dictionary {algorithm_name: episode_lengths}
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    for (algo, episodes), color in zip(results_dict.items(), colors):
        # Smooth the data
        window = min(50, len(episodes) // 10)
        if len(episodes) > window:
            smoothed = np.convolve(episodes, np.ones(window)/window, 
                                  mode='valid')
            plt.plot(smoothed, label=algo, color=color, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_q_function(Q, env, action_names=None, title="Q-Function Heatmap"):
    """
    Plot Q-function as heatmap for each action
    
    Args:
        Q: Q-function (states x actions)
        env: Environment object
        action_names: Names for actions
        title: Plot title
    """
    if action_names is None:
        action_names = ['Up', 'Right', 'Down', 'Left']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    for action, ax, name in zip(range(4), axes.flatten(), action_names):
        Q_action = Q[:, action].reshape(env.height, env.width)
        
        sns.heatmap(Q_action, annot=True, fmt='.1f', cmap='coolwarm',
                   center=0, ax=ax, cbar_kws={'label': 'Q-value'})
        
        # Mark special positions
        ax.scatter(env.start_pos[1] + 0.5, env.start_pos[0] + 0.5,
                  marker='s', s=100, c='green')
        ax.scatter(env.goal_pos[1] + 0.5, env.goal_pos[0] + 0.5,
                  marker='*', s=150, c='gold')
        
        ax.set_title(f'Action: {name}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    plt.tight_layout()
    return fig


def create_convergence_plot(value_history, title="Value Function Convergence"):
    """
    Plot convergence of value function over iterations
    
    Args:
        value_history: List of value functions at each iteration
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate max change at each iteration
    max_changes = []
    for i in range(1, len(value_history)):
        max_change = np.max(np.abs(value_history[i] - value_history[i-1]))
        max_changes.append(max_change)
    
    plt.semilogy(range(1, len(max_changes) + 1), max_changes)
    plt.xlabel('Iteration')
    plt.ylabel('Max Change in Value Function (log scale)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()