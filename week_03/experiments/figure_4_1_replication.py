"""
Replication of Sutton & Barto Figure 4.1
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from gridworld import Gridworld


def iterative_policy_evaluation(gridworld, policy, theta=1e-4, gamma=1.0):
    """
    Iterative policy evaluation algorithm
    
    Args:
        gridworld: Gridworld environment
        policy: Policy to evaluate (uniform random)
        theta: Convergence threshold
        gamma: Discount factor
        
    Returns:
        List of value functions at each iteration
    """
    V = np.zeros(gridworld.n_states)
    history = [V.copy()]
    
    iteration = 0
    while True:
        delta = 0
        V_new = np.zeros(gridworld.n_states)
        
        for state in range(gridworld.n_states):
            if gridworld.is_terminal(state):
                continue
                
            v = 0
            for action in range(gridworld.n_actions):
                # For each action, calculate expected value
                next_state = gridworld.get_next_state(state, action)
                reward = gridworld.get_reward(state, action, next_state)
                
                # Uniform random policy
                action_prob = policy[state, action]
                v += action_prob * (reward + gamma * V[next_state])
                
            V_new[state] = v
            delta = max(delta, abs(V[state] - V_new[state]))
            
        V = V_new.copy()
        iteration += 1
        
        # Store snapshots at specific iterations
        if iteration in [1, 2, 3, 10, 1000] or (iteration < 10):
            history.append(V.copy())
            print(f"Iteration {iteration}: Delta = {delta:.6f}")
            
        if delta < theta:
            print(f"Converged at iteration {iteration}")
            # Add final converged value
            history.append(V.copy())
            break
            
    return history


def replicate_figure_4_1():
    """
    Main function to replicate Figure 4.1
    """
    print("=" * 60)
    print("Replicating Sutton & Barto Figure 4.1")
    print("4x4 Gridworld with Iterative Policy Evaluation")
    print("=" * 60)
    
    # Create gridworld
    gridworld = Gridworld(size=4)
    
    # Create uniform random policy
    policy = np.ones((gridworld.n_states, gridworld.n_actions)) / gridworld.n_actions
    
    # Run iterative policy evaluation
    print("\nRunning Iterative Policy Evaluation...")
    value_history = iterative_policy_evaluation(gridworld, policy)
    
    # Display results at key iterations
    key_iterations = [0, 1, 2, 3, 10, -1]  # -1 for converged
    iteration_labels = ['k=0', 'k=1', 'k=2', 'k=3', 'k=10', 'k=∞']
    
    print("\n" + "=" * 60)
    print("Value Functions at Key Iterations")
    print("=" * 60)
    
    for idx, k in enumerate(key_iterations):
        if k == -1:
            V = value_history[-1]
            label = iteration_labels[idx]
        else:
            # Find the value function for this iteration
            if k < len(value_history):
                V = value_history[k]
            else:
                V = value_history[-1]
            label = iteration_labels[idx]
            
        print(f"\n{label}:")
        print("-" * 30)
        
        # Display as grid
        for row in range(gridworld.size):
            row_str = ""
            for col in range(gridworld.size):
                state = gridworld.position_to_state(row, col)
                if gridworld.is_terminal(state):
                    row_str += "  0.0 "
                else:
                    row_str += f"{V[state]:6.1f}"
                    
                if col < gridworld.size - 1:
                    row_str += " | "
            print(row_str)
            
    # Create visualization similar to the figure
    create_figure_visualization(gridworld, value_history, key_iterations, iteration_labels)
    
    return value_history


def create_figure_visualization(gridworld, value_history, key_iterations, labels):
    """
    Create visualization similar to Figure 4.1
    
    Args:
        gridworld: Gridworld environment
        value_history: List of value functions
        key_iterations: Which iterations to show
        labels: Labels for each iteration
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Figure 4.1 Replication: Iterative Policy Evaluation', fontsize=14)
    
    axes = axes.flatten()
    
    for idx, (k, label) in enumerate(zip(key_iterations, labels)):
        ax = axes[idx]
        
        if k == -1:
            V = value_history[-1]
        else:
            if k < len(value_history):
                V = value_history[k]
            else:
                V = value_history[-1]
                
        # Reshape values for display
        V_grid = np.zeros((gridworld.size, gridworld.size))
        for row in range(gridworld.size):
            for col in range(gridworld.size):
                state = gridworld.position_to_state(row, col)
                V_grid[row, col] = V[state]
                
        # Create heatmap
        im = ax.imshow(V_grid, cmap='coolwarm', vmin=-22, vmax=0)
        
        # Add value text
        for row in range(gridworld.size):
            for col in range(gridworld.size):
                state = gridworld.position_to_state(row, col)
                if gridworld.is_terminal(state):
                    ax.text(col, row, '0.0', ha='center', va='center',
                           color='black', fontweight='bold')
                else:
                    ax.text(col, row, f'{V[state]:.1f}', ha='center', va='center',
                           color='white' if V[state] < -11 else 'black')
                    
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Mark terminal states
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, 
                                  edgecolor='green', linewidth=3))
        ax.add_patch(plt.Rectangle((gridworld.size-1.5, gridworld.size-1.5), 1, 1, 
                                  fill=False, edgecolor='green', linewidth=3))
    
    # Add colorbar
    plt.colorbar(im, ax=axes, label='Value', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('figure_4_1_replication.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nFigure saved as 'figure_4_1_replication.png'")


def display_greedy_policy(gridworld, V):
    """
    Display the greedy policy with respect to value function
    
    Args:
        gridworld: Gridworld environment
        V: Value function
    """
    print("\n" + "=" * 60)
    print("Greedy Policy (Arrows indicate best actions)")
    print("=" * 60)
    
    policy = np.zeros((gridworld.n_states, gridworld.n_actions))
    
    for state in range(gridworld.n_states):
        if gridworld.is_terminal(state):
            continue
            
        action_values = []
        for action in range(gridworld.n_actions):
            next_state = gridworld.get_next_state(state, action)
            reward = gridworld.get_reward(state, action, next_state)
            action_values.append(reward + V[next_state])
            
        # Make greedy policy
        best_value = max(action_values)
        best_actions = [a for a, v in enumerate(action_values) if v == best_value]
        
        for a in best_actions:
            policy[state, a] = 1.0 / len(best_actions)
            
    gridworld.display_policy(policy)
    
    return policy


if __name__ == "__main__":
    # Run the replication
    value_history = replicate_figure_4_1()
    
    # Also show the optimal policy
    gridworld = Gridworld(size=4)
    final_V = value_history[-1]
    
    greedy_policy = display_greedy_policy(gridworld, final_V)
    
    print("\nReplication complete!")
    print("This demonstrates the convergence of iterative policy evaluation")
    print("for a uniform random policy on the 4x4 gridworld.")
    print("\nNote: The greedy policy shown would be the result of")
    print("policy improvement if we were doing policy iteration.")