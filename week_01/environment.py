
"""
Created on Sep 21 02:04:22 2025

@author: mehar

This module defines the bandit environment for the reinforcement learning experiments.

"""

import numpy as np


class BanditEnvironment:
    """
    K-Armed Bandit Environment
    
    This class represents a stationary multi-armed bandit problem where
    each arm has a fixed true reward value sampled from a normal distribution.
    """
    
    def __init__(self, n_arms=10, random_seed=None):
        """
        Initialize the bandit environment.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit (default: 10)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.n_arms = n_arms
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize true action values q*(a) from standard normal distribution
        self.true_values = np.random.normal(0, 1, n_arms)
        
        # Identify the optimal action
        self.best_action = np.argmax(self.true_values)
        
        # Track statistics
        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms)
    
    def pull_arm(self, action):
        """
        Pull a specific arm and receive a reward.
        
        The reward is the true value of the arm plus Gaussian noise.
        
        Parameters:
        -----------
        action : int
            Index of the arm to pull (0 to n_arms-1)
        
        Returns:
        --------
        float
            Reward received from pulling the arm
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.n_arms-1}")
        
        # Generate reward: true value + noise
        reward = self.true_values[action] + np.random.normal(0, 1)
        
        # Update statistics
        self.total_pulls += 1
        self.arm_pulls[action] += 1
        
        return reward
    
    def is_optimal_action(self, action):
        """
        Check if the given action is optimal.
        
        Parameters:
        -----------
        action : int
            Action to check
        
        Returns:
        --------
        bool
            True if action is optimal, False otherwise
        """
        return action == self.best_action
    
    def get_optimal_value(self):
        """
        Get the true value of the optimal action.
        
        Returns:
        --------
        float
            Value of the best arm
        """
        return self.true_values[self.best_action]
    
    def reset_statistics(self):
        """Reset pull statistics while keeping the same bandit values."""
        self.total_pulls = 0
        self.arm_pulls = np.zeros(self.n_arms)
    
    def get_statistics(self):
        """
        Get current statistics about arm pulls.
        
        Returns:
        --------
        dict
            Dictionary containing pull statistics
        """
        return {
            'total_pulls': self.total_pulls,
            'arm_pulls': self.arm_pulls.copy(),
            'optimal_action': self.best_action,
            'optimal_value': self.get_optimal_value()
        }

