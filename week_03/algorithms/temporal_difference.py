"""
Temporal Difference Learning Algorithms
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import List, Tuple
import random
import sys
sys.path.append('..')
from windy_gridworld import WindyGridworld


class TemporalDifference:
    """
    TD Learning algorithms for Windy Gridworld
    """
    
    def __init__(self, env: WindyGridworld, gamma: float = 0.9):
        """
        Initialize TD learner
        
        Args:
            env: Windy Gridworld environment
            gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma
        self.n_states = env.height * env.width  
        self.n_actions = env.n_actions
        
        # Initialize Q-function
        self.Q = np.zeros((self.n_states, self.n_actions))
        
    def sarsa(self, n_episodes: int = 500, alpha: float = 0.5, 
              epsilon: float = 0.1) -> Tuple[np.ndarray, List[int]]:
        """
        SARSA: On-policy TD(0) control
        
        Args:
            n_episodes: Number of episodes
            alpha: Learning rate
            epsilon: Exploration rate
            
        Returns:
            (Q-function, episode_lengths)
        """
        print(f"\nRunning SARSA (TD(0) On-Policy Control)...")
        print(f"Episodes: {n_episodes}, Alpha: {alpha}, Epsilon: {epsilon}")
        print("=" * 60)
        
        episode_lengths = []
        
        for episode in range(n_episodes):
            # Initialize S
            pos = self.env.reset()
            state = self.env.position_to_state(pos)
            
            # Choose A from S using epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = np.argmax(self.Q[state])
                
            steps = 0
            
            while True:
                # Take action A, observe R, S'
                next_pos, reward, done = self.env.step(action)
                next_state = self.env.position_to_state(next_pos)
                steps += 1
                
                if not done:
                    # Choose A' from S' using epsilon-greedy
                    if random.random() < epsilon:
                        next_action = random.randint(0, self.n_actions - 1)
                    else:
                        next_action = np.argmax(self.Q[next_state])
                        
                    # TD update
                    td_target = reward + self.gamma * self.Q[next_state, next_action]
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += alpha * td_error
                    
                    # S <- S', A <- A'
                    state = next_state
                    action = next_action
                else:
                    # Terminal state update
                    td_target = reward
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += alpha * td_error
                    break
                    
                if steps >= 10000:
                    print(f"Episode {episode} exceeded max steps")
                    break
                    
            episode_lengths.append(steps)
            
            # Progress reporting
            if (episode + 1) % 100 == 0:
                avg_length = np.mean(episode_lengths[-50:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg length (last 50): {avg_length:.1f}")
                
        return self.Q, episode_lengths
    
    def td_off_policy_unweighted_is(self, n_episodes: int = 500, 
                                    alpha: float = 0.5,
                                    behavior_epsilon: float = 0.1) -> Tuple[np.ndarray, List[int]]:
        """
        TD(0) Off-Policy Control with Unweighted Importance Sampling
        
        Args:
            n_episodes: Number of episodes
            alpha: Learning rate
            behavior_epsilon: Exploration for behavior policy
            
        Returns:
            (Q-function, episode_lengths)
        """
        print(f"\nRunning TD(0) Off-Policy with Unweighted Importance Sampling...")
        print(f"Episodes: {n_episodes}, Alpha: {alpha}, Epsilon: {behavior_epsilon}")
        print("=" * 60)
        
        episode_lengths = []
        
        for episode in range(n_episodes):
            pos = self.env.reset()
            state = self.env.position_to_state(pos)
            
            # Choose initial action from behavior policy (epsilon-greedy)
            if random.random() < behavior_epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = np.argmax(self.Q[state])
            
            steps = 0
            
            while True:
                # Take action
                next_pos, reward, done = self.env.step(action)
                next_state = self.env.position_to_state(next_pos)
                steps += 1
                
                # Calculate importance sampling ratio
                # Target policy: greedy
                target_action = np.argmax(self.Q[state])
                if action == target_action:
                    # Calculate behavior policy probability
                    if action == np.argmax(self.Q[state]):
                        behavior_prob = 1 - behavior_epsilon + behavior_epsilon / self.n_actions
                    else:
                        behavior_prob = behavior_epsilon / self.n_actions
                    # Target policy is deterministic (prob = 1 for best action)
                    importance_ratio = 1.0 / behavior_prob
                else:
                    # Target policy wouldn't take this action
                    importance_ratio = 0.0
                
                if not done:
                    # Choose next action from behavior policy
                    if random.random() < behavior_epsilon:
                        next_action = random.randint(0, self.n_actions - 1)
                    else:
                        next_action = np.argmax(self.Q[next_state])
                    
                    # TD update with importance sampling
                    td_target = reward + self.gamma * self.Q[next_state, next_action]
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += alpha * importance_ratio * td_error
                    
                    state = next_state
                    action = next_action
                else:
                    # Terminal state update
                    td_target = reward
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += alpha * importance_ratio * td_error
                    break
                    
                if steps >= 10000:
                    break
                    
            episode_lengths.append(steps)
            
            if (episode + 1) % 100 == 0:
                avg_length = np.mean(episode_lengths[-50:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg length (last 50): {avg_length:.1f}")
                
        return self.Q, episode_lengths
    
    def td_off_policy_weighted_is(self, n_episodes: int = 500,
                                 alpha: float = 0.5,
                                 behavior_epsilon: float = 0.1) -> Tuple[np.ndarray, List[int]]:
        """
        TD(0) Off-Policy Control with Weighted Importance Sampling
        
        Args:
            n_episodes: Number of episodes
            alpha: Learning rate
            behavior_epsilon: Exploration for behavior policy
            
        Returns:
            (Q-function, episode_lengths)
        """
        print(f"\nRunning TD(0) Off-Policy with Weighted Importance Sampling...")
        print(f"Episodes: {n_episodes}, Alpha: {alpha}, Epsilon: {behavior_epsilon}")
        print("=" * 60)
        
        # Track cumulative importance sampling weights
        C = np.zeros((self.n_states, self.n_actions))
        episode_lengths = []
        
        for episode in range(n_episodes):
            pos = self.env.reset()
            state = self.env.position_to_state(pos)
            
            # Choose initial action from behavior policy
            if random.random() < behavior_epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = np.argmax(self.Q[state])
            
            steps = 0
            
            while True:
                # Take action
                next_pos, reward, done = self.env.step(action)
                next_state = self.env.position_to_state(next_pos)
                steps += 1
                
                # Calculate importance sampling ratio
                target_action = np.argmax(self.Q[state])
                if action == target_action:
                    if action == np.argmax(self.Q[state]):
                        behavior_prob = 1 - behavior_epsilon + behavior_epsilon / self.n_actions
                    else:
                        behavior_prob = behavior_epsilon / self.n_actions
                    importance_ratio = 1.0 / behavior_prob
                else:
                    importance_ratio = 0.0
                
                # Update cumulative weights
                C[state, action] += importance_ratio
                
                if not done:
                    # Choose next action from behavior policy
                    if random.random() < behavior_epsilon:
                        next_action = random.randint(0, self.n_actions - 1)
                    else:
                        next_action = np.argmax(self.Q[next_state])
                    
                    # Weighted TD update
                    td_target = reward + self.gamma * self.Q[next_state, next_action]
                    td_error = td_target - self.Q[state, action]
                    
                    if C[state, action] > 0:
                        weighted_alpha = alpha * importance_ratio / C[state, action]
                        self.Q[state, action] += weighted_alpha * td_error
                    
                    state = next_state
                    action = next_action
                else:
                    # Terminal state update
                    td_target = reward
                    td_error = td_target - self.Q[state, action]
                    
                    if C[state, action] > 0:
                        weighted_alpha = alpha * importance_ratio / C[state, action]
                        self.Q[state, action] += weighted_alpha * td_error
                    break
                    
                if steps >= 10000:
                    break
                    
            episode_lengths.append(steps)
            
            if (episode + 1) % 100 == 0:
                avg_length = np.mean(episode_lengths[-50:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg length (last 50): {avg_length:.1f}")
                
        return self.Q, episode_lengths
    
    def extract_policy_from_q(self, Q: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Extract policy from Q-function
        
        Args:
            Q: Q-function
            epsilon: Epsilon for epsilon-greedy policy (0 for greedy)
            
        Returns:
            Policy array
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for state in range(self.n_states):
            if epsilon == 0:
                # Greedy policy
                best_action = np.argmax(Q[state])
                policy[state, best_action] = 1.0
            else:
                # Epsilon-greedy
                best_action = np.argmax(Q[state])
                for action in range(self.n_actions):
                    if action == best_action:
                        policy[state, action] = 1 - epsilon + epsilon / self.n_actions
                    else:
                        policy[state, action] = epsilon / self.n_actions
                        
        return policy