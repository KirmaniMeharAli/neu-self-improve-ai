"""
Monte Carlo Control Algorithms
Author: Mehar  
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import List, Tuple, Dict
import random
import sys
sys.path.append('..')
from windy_gridworld import WindyGridworld


class MonteCarloControl:
    """
    Monte Carlo control algorithms for Windy Gridworld
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 0.9):
        """
        Initialize MC controller
        
        Args:
            env: Windy Gridworld environment
            gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma
        self.n_states = env.height * env.width
        self.n_actions = env.n_actions
        
        # Initialize Q-function and policy
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
        # Visit counts for averaging
        self.returns_sum = np.zeros((self.n_states, self.n_actions))
        self.returns_count = np.zeros((self.n_states, self.n_actions))
        
        # For importance sampling
        self.C = np.zeros((self.n_states, self.n_actions))  # Cumulative sum of weights
        
    def generate_episode(self, policy: np.ndarray, epsilon: float = 0.1, 
                         max_steps: int = 1000) -> List[Tuple[int, int, float]]:
        """
        Generated an episode using epsilon-greedy policy
        
        Args:
            policy: Current policy
            epsilon: Exploration parameter
            max_steps: Maximum episode length
            
        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        pos = self.env.reset()
        
        for step in range(max_steps):
            state_idx = self.env.position_to_state(pos)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                # Use provided policy
                action_probs = policy[state_idx]
                action = np.random.choice(self.n_actions, p=action_probs)
            
            # Take action
            next_pos, reward, done = self.env.step(action)
            episode.append((state_idx, action, reward))
            
            if done:
                break
                
            pos = next_pos
            
        return episode
    
    def on_policy_mc_control(self, n_episodes: int = 10000, 
                             epsilon: float = 0.1) -> Tuple[np.ndarray, List[float]]:
        """
        On-policy first-visit MC control with epsilon-soft policies
        
        Args:
            n_episodes: Number of episodes to run
            epsilon: Exploration parameter
            
        Returns:
            (Q-function, episode_lengths)
        """
        print(f"\nRunning On-Policy Monte Carlo Control...")
        print(f"Episodes: {n_episodes}, Epsilon: {epsilon}")
        print("=" * 60)
        
        episode_lengths = []
        
        for episode_num in range(n_episodes):
            # Generate episode
            episode = self.generate_episode(self.policy, epsilon)
            episode_lengths.append(len(episode))
            
            # Calculate returns
            G = 0
            visited_state_actions = set()
            
            # Process episode backwards
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                
                G = self.gamma * G + reward
                
                # First-visit check
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    # Incremental average
                    self.returns_sum[state, action] += G
                    self.returns_count[state, action] += 1
                    self.Q[state, action] = (self.returns_sum[state, action] / 
                                           self.returns_count[state, action])
                    
                    # Update policy to be epsilon-greedy
                    best_action = np.argmax(self.Q[state])
                    
                    for a in range(self.n_actions):
                        if a == best_action:
                            self.policy[state, a] = 1 - epsilon + epsilon / self.n_actions
                        else:
                            self.policy[state, a] = epsilon / self.n_actions
            
            # Progress reporting
            if (episode_num + 1) % 1000 == 0:
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode_num + 1}/{n_episodes}, "
                      f"Avg length (last 100): {avg_length:.1f}")
                
        return self.Q, episode_lengths
    
    def off_policy_mc_control(self, n_episodes: int = 10000,
                             behavior_epsilon: float = 0.1) -> Tuple[np.ndarray, List[float]]:
        """
        Off-policy MC control using importance sampling
        
        Args:
            n_episodes: Number of episodes  
            behavior_epsilon: Exploration for behavior policy
            
        Returns:
            (Q-function, episode_lengths)
        """
        print(f"\nRunning Off-Policy Monte Carlo Control...")
        print(f"Episodes: {n_episodes}, Behavior Epsilon: {behavior_epsilon}")
        print("=" * 60)
        
        # Initialize target policy (deterministic)
        target_policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            target_policy[s, 0] = 1.0  # Start with all UP
            
        # Behavior policy (epsilon-greedy)
        behavior_policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
        episode_lengths = []
        
        for episode_num in range(n_episodes):
            # Generate episode using behavior policy
            episode = []
            pos = self.env.reset()
            
            for _ in range(1000):
                state_idx = self.env.position_to_state(pos)
                
                # Use behavior policy (epsilon-greedy)
                if random.random() < behavior_epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    action = np.argmax(self.Q[state_idx])
                    
                next_pos, reward, done = self.env.step(action)
                episode.append((state_idx, action, reward))
                
                if done:
                    break
                pos = next_pos
                
            episode_lengths.append(len(episode))
            
            # Update Q using importance sampling
            G = 0
            W = 1.0
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                
                G = self.gamma * G + reward
                
                # Update cumulative importance sampling
                self.C[state, action] += W
                self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
                
                # Update target policy
                target_policy[state] = np.zeros(self.n_actions)
                target_policy[state, np.argmax(self.Q[state])] = 1.0
                
                # If action doesn't match target policy, importance ratio is 0
                if action != np.argmax(target_policy[state]):
                    break
                    
                # Update importance ratio
                behavior_prob = (behavior_epsilon / self.n_actions + 
                                (1 - behavior_epsilon) if action == np.argmax(self.Q[state]) 
                                else behavior_epsilon / self.n_actions)
                W = W / behavior_prob
            
            # Progress reporting
            if (episode_num + 1) % 1000 == 0:
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode_num + 1}/{n_episodes}, "
                      f"Avg length (last 100): {avg_length:.1f}")
                
        # Set final policy
        self.policy = target_policy
        
        return self.Q, episode_lengths