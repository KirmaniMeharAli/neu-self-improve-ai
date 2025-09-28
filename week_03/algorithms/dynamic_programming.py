"""
Dynamic Programming Algorithms
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import Tuple, Optional
import sys
sys.path.append('..')
from windy_gridworld import WindyGridworld


class DynamicProgramming:
    """
    Dynamic Programming algorithms for gridworld
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 0.9):
        """
        Initialize DP solver
        
        Args:
            env: Windy Gridworld environment
            gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma
        self.n_states = env.height * env.width
        self.n_actions = env.n_actions
        
        # Initialize value function and policy
        self.V = np.zeros(self.n_states)
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-6) -> np.ndarray:
        """
        Evaluate a given policy
        
        Args:
            policy: Policy to evaluate (states x actions)
            theta: Convergence threshold
            
        Returns:
            Value function for the policy
        """
        V = np.zeros(self.n_states)
        iteration = 0
        
        while True:
            delta = 0
            iteration += 1
            
            # Update each state
            for state_idx in range(self.n_states):
                pos = self.env.state_to_position(state_idx)
                
                if self.env.is_terminal(pos):
                    continue
                    
                v = V[state_idx]
                new_v = 0
                
                # Consider all actions
                for action in range(self.n_actions):
                    action_value = 0
                    
                    # Get transition probabilities
                    transitions = self.env.get_transition_probabilities(pos, action)
                    
                    for next_pos, prob in transitions.items():
                        next_state_idx = self.env.position_to_state(next_pos)
                        
                        # Calculate reward
                        if self.env.is_terminal(next_pos):
                            reward = 0
                        else:
                            reward = -1
                            
                        action_value += prob * (reward + self.gamma * V[next_state_idx])
                    
                    new_v += policy[state_idx, action] * action_value
                
                V[state_idx] = new_v
                delta = max(delta, abs(v - V[state_idx]))
            
            print(f"Policy Evaluation - Iteration {iteration}, Delta: {delta:.8f}")
            
            if delta < theta:
                break
                
        return V
    
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Improve policy based on value function
        
        Args:
            V: Current value function
            
        Returns:
            (new_policy, policy_stable)
        """
        new_policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for state_idx in range(self.n_states):
            pos = self.env.state_to_position(state_idx)
            
            if self.env.is_terminal(pos):
                # Terminal state - doesn't matter
                new_policy[state_idx] = np.ones(self.n_actions) / self.n_actions
                continue
            
            # Find best action(s)
            action_values = np.zeros(self.n_actions)
            
            for action in range(self.n_actions):
                transitions = self.env.get_transition_probabilities(pos, action)
                
                for next_pos, prob in transitions.items():
                    next_state_idx = self.env.position_to_state(next_pos)
                    
                    if self.env.is_terminal(next_pos):
                        reward = 0
                    else:
                        reward = -1
                        
                    action_values[action] += prob * (reward + self.gamma * V[next_state_idx])
            
            # Create deterministic policy choosing best action
            best_actions = np.argwhere(action_values == np.max(action_values)).flatten()
            
            # Check if policy changed
            old_best = np.argwhere(self.policy[state_idx] > 0).flatten()
            if not np.array_equal(sorted(best_actions), sorted(old_best)):
                policy_stable = False
            
            # Update policy (deterministic)
            new_policy[state_idx] = np.zeros(self.n_actions)
            new_policy[state_idx, best_actions] = 1.0 / len(best_actions)
            
        return new_policy, policy_stable
    
    def policy_iteration(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Policy Iteration algorithm
        
        Args:
            theta: Convergence threshold for evaluation
            
        Returns:
            (optimal_value_function, optimal_policy)
        """
        print("\nStarting Policy Iteration...")
        print("=" * 60)
        
        # Start with random policy
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n--- Policy Iteration Round {iteration} ---")
            
            # Policy Evaluation
            self.V = self.policy_evaluation(self.policy, theta)
            
            # Policy Improvement
            self.policy, policy_stable = self.policy_improvement(self.V)
            
            if policy_stable:
                print(f"\nPolicy stable after {iteration} iterations!")
                break
                
        return self.V, self.policy
    
    def value_iteration(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Value Iteration algorithm
        
        Args:
            theta: Convergence threshold
            
        Returns:
            (optimal_value_function, optimal_policy)
        """
        print("\nStarting Value Iteration...")
        print("=" * 60)
        
        V = np.zeros(self.n_states)
        iteration = 0
        
        while True:
            delta = 0
            iteration += 1
            
            for state_idx in range(self.n_states):
                pos = self.env.state_to_position(state_idx)
                
                if self.env.is_terminal(pos):
                    continue
                    
                v = V[state_idx]
                
                # Calculate value for each action
                action_values = []
                for action in range(self.n_actions):
                    action_value = 0
                    transitions = self.env.get_transition_probabilities(pos, action)
                    
                    for next_pos, prob in transitions.items():
                        next_state_idx = self.env.position_to_state(next_pos)
                        
                        if self.env.is_terminal(next_pos):
                            reward = 0
                        else:
                            reward = -1
                            
                        action_value += prob * (reward + self.gamma * V[next_state_idx])
                    
                    action_values.append(action_value)
                
                # Take maximum over actions
                V[state_idx] = max(action_values)
                delta = max(delta, abs(v - V[state_idx]))
            
            print(f"Value Iteration - Iteration {iteration}, Delta: {delta:.8f}")
            
            if delta < theta:
                break
        
        # Extract policy from value function
        policy = np.zeros((self.n_states, self.n_actions))
        
        for state_idx in range(self.n_states):
            pos = self.env.state_to_position(state_idx)
            
            if self.env.is_terminal(pos):
                policy[state_idx] = np.ones(self.n_actions) / self.n_actions
                continue
            
            action_values = np.zeros(self.n_actions)
            
            for action in range(self.n_actions):
                transitions = self.env.get_transition_probabilities(pos, action)
                
                for next_pos, prob in transitions.items():
                    next_state_idx = self.env.position_to_state(next_pos)
                    
                    if self.env.is_terminal(next_pos):
                        reward = 0
                    else:
                        reward = -1
                        
                    action_values[action] += prob * (reward + self.gamma * V[next_state_idx])
            
            best_action = np.argmax(action_values)
            policy[state_idx, best_action] = 1.0
            
        print(f"\nValue Iteration converged after {iteration} iterations!")
        
        self.V = V
        self.policy = policy
        
        return V, policy