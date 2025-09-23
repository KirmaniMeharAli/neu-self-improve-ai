"""
Created on Sep 22 02:04:22 2025

@author: mehar

This module contains different agent strategies for solving the multi-armed bandit problem.

"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all bandit agents.
    """
    
    def __init__(self, n_arms):
        """
        Initialize base agent.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit
        """
        self.n_arms = n_arms
        self.t = 0  # Time step
        
    @abstractmethod
    def select_action(self):
        """Select an action to take."""
        pass
    
    @abstractmethod
    def update(self, action, reward):
        """Update agent's internal state based on observed reward."""
        pass
    
    def reset(self):
        """Reset agent to initial state."""
        self.t = 0


class EpsilonGreedyAgent(BaseAgent):
    """
    Epsilon-Greedy Agent
    
    This agent balances exploration and exploitation using the epsilon-greedy strategy.
    With probability epsilon, it explores by choosing a random action.
    Otherwise, it exploits by choosing the action with highest estimated value.
    """
    
    def __init__(self, n_arms, epsilon=0.1, initial_values=0.0):
        """
        Initialize epsilon-greedy agent.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms
        epsilon : float
            Exploration probability (0 <= epsilon <= 1)
        initial_values : float
            Initial value estimates for all arms
        """
        super().__init__(n_arms)
        self.epsilon = epsilon
        
        # Initialize action value estimates
        self.q_estimates = np.ones(n_arms) * initial_values
        
        # Track number of times each action has been selected
        self.action_counts = np.zeros(n_arms)
    
    def select_action(self):
        """
        Select action using epsilon-greedy strategy.
        
        Returns:
        --------
        int
            Selected action
        """
        self.t += 1
        
        # Exploration vs exploitation
        if np.random.random() < self.epsilon:
            # Explore: select random action
            return np.random.randint(self.n_arms)
        else:
            # Exploit: select greedy action (with random tie-breaking)
            max_value = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_value)[0]
            return np.random.choice(best_actions)
    
    def update(self, action, reward):
        """
        Update action value estimate using incremental update rule.
        
        Parameters:
        -----------
        action : int
            Action that was taken
        reward : float
            Observed reward
        """
        # Increment action count
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update: Q_new = Q_old + (1/n) * (R - Q_old)
        old_estimate = self.q_estimates[action]
        self.q_estimates[action] = old_estimate + (1.0 / n) * (reward - old_estimate)
    
    def reset(self):
        """Reset agent to initial state."""
        super().reset()
        self.q_estimates = np.zeros(self.n_arms)
        self.action_counts = np.zeros(self.n_arms)


class GradientBanditAgent(BaseAgent):
    """
    Gradient Bandit Agent
    
    This agent learns a numerical preference for each action using gradient ascent.
    Actions are selected according to a softmax distribution over preferences.
    """
    
    def __init__(self, n_arms, learning_rate=0.1, use_baseline=True):
        """
        Initialize gradient bandit agent.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms
        learning_rate : float
            Step size parameter (alpha) for gradient updates
        use_baseline : bool
            Whether to use average reward as baseline
        """
        super().__init__(n_arms)
        self.alpha = learning_rate
        self.use_baseline = use_baseline
        
        # Initialize action preferences (H values)
        self.preferences = np.zeros(n_arms)
        
        # Baseline (average reward)
        self.average_reward = 0.0
        
        # Store last action probabilities for efficiency
        self.last_action_probs = None
    
    def get_action_probabilities(self):
        """
        Convert preferences to action probabilities using softmax.
        
        Returns:
        --------
        numpy.ndarray
            Probability distribution over actions
        """
        # Numerical stability: subtract max preference
        max_pref = np.max(self.preferences)
        exp_preferences = np.exp(self.preferences - max_pref)
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        return probabilities
    
    def select_action(self):
        """
        Select action according to softmax distribution.
        
        Returns:
        --------
        int
            Selected action
        """
        self.t += 1
        
        # Get action probabilities
        self.last_action_probs = self.get_action_probabilities()
        
        # Sample action from probability distribution
        action = np.random.choice(self.n_arms, p=self.last_action_probs)
        
        return action
    
    def update(self, action, reward):
        """
        Update preferences using gradient ascent.
        
        Parameters:
        -----------
        action : int
            Action that was taken
        reward : float
            Observed reward
        """
        # Update baseline (incremental average)
        if self.use_baseline:
            self.average_reward += (reward - self.average_reward) / self.t
            baseline = self.average_reward
        else:
            baseline = 0.0
        
        # Use stored probabilities or recalculate if needed
        if self.last_action_probs is None:
            action_probs = self.get_action_probabilities()
        else:
            action_probs = self.last_action_probs
        
        # Gradient ascent update for all preferences
        for a in range(self.n_arms):
            if a == action:
                # For selected action: H(a) += α * (R - baseline) * (1 - π(a))
                self.preferences[a] += self.alpha * (reward - baseline) * (1 - action_probs[a])
            else:
                # For other actions: H(a) -= α * (R - baseline) * π(a)
                self.preferences[a] -= self.alpha * (reward - baseline) * action_probs[a]
        
        # Clear stored probabilities
        self.last_action_probs = None
    
    def reset(self):
        """Reset agent to initial state."""
        super().reset()
        self.preferences = np.zeros(self.n_arms)
        self.average_reward = 0.0
        self.last_action_probs = None


class OptimalAgent(BaseAgent):
    """
    Optimal Agent (for comparison purposes)
    
    This agent always selects the optimal action. Used as a baseline
    to compare other algorithms against perfect performance.
    """
    
    def __init__(self, n_arms, optimal_action):
        """
        Initialize optimal agent.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms
        optimal_action : int
            Index of the optimal action
        """
        super().__init__(n_arms)
        self.optimal_action = optimal_action
    
    def select_action(self):
        """Always select the optimal action."""
        self.t += 1
        return self.optimal_action
    
    def update(self, action, reward):
        """No learning needed for optimal agent."""
        pass