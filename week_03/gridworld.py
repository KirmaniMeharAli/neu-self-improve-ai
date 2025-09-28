"""
Basic Gridworld Implementation
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import Tuple, Dict, List


class Gridworld:
    """
    Implementation of the 4x4 gridworld from Example 4.1
    """
    
    def __init__(self, size: int = 4):
        """
        Initialize the gridworld environment
        
        Args:
            size: Grid dimension (creates size x size grid)
        """
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        
        # Define terminal states (corners)
        self.terminal_states = [0, self.n_states - 1]
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        self.action_symbols = {
            0: '↑',
            1: '→',
            2: '↓',
            3: '←'
        }
        
        # Initialize value function
        self.reset_values()
        
    def reset_values(self):
        """Reset value function to zeros"""
        self.V = np.zeros(self.n_states)
        
    def state_to_position(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position
        
        Args:
            state: State index
            
        Returns:
            (row, col) position
        """
        row = state // self.size
        col = state % self.size
        return row, col
    
    def position_to_state(self, row: int, col: int) -> int:
        """
        Convert grid position to state index
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            State index
        """
        return row * self.size + col
    
    def is_terminal(self, state: int) -> bool:
        """Check if state is terminal"""
        return state in self.terminal_states
    
    def get_next_state(self, state: int, action: int) -> int:
        """
        Get next state given current state and action
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state
        """
        if self.is_terminal(state):
            return state
            
        row, col = self.state_to_position(state)
        delta_row, delta_col = self.actions[action]
        
        # Calculate new position
        new_row = row + delta_row
        new_col = col + delta_col
        
        # Check boundaries
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            return self.position_to_state(new_row, new_col)
        else:
            # Stay in same state if hit boundary
            return state
    
    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        if self.is_terminal(state):
            return 0.0
        return -1.0
    
    def get_transition_prob(self, state: int, action: int, next_state: int) -> float:
        """
        Get transition probability (deterministic for basic gridworld)
        
        Args:
            state: Current state
            action: Action taken
            next_state: Target next state
            
        Returns:
            Transition probability
        """
        expected_next = self.get_next_state(state, action)
        return 1.0 if next_state == expected_next else 0.0
    
    def display_values(self):
        """Display value function as grid"""
        print("\nValue Function:")
        print("-" * 50)
        
        for row in range(self.size):
            row_str = ""
            for col in range(self.size):
                state = self.position_to_state(row, col)
                if self.is_terminal(state):
                    row_str += "  T  "
                else:
                    row_str += f"{self.V[state]:5.1f}"
                row_str += " | " if col < self.size - 1 else ""
            print(row_str)
            
            if row < self.size - 1:
                print("-" * 50)
    
    def display_policy(self, policy: np.ndarray):
        """
        Display policy as grid with arrows
        
        Args:
            policy: Policy array (states x actions)
        """
        print("\nPolicy (Greedy actions):")
        print("-" * 50)
        
        for row in range(self.size):
            row_str = ""
            for col in range(self.size):
                state = self.position_to_state(row, col)
                
                if self.is_terminal(state):
                    row_str += "  T  "
                else:
                    # Find best action(s)
                    best_actions = np.where(policy[state] == policy[state].max())[0]
                    
                    if len(best_actions) == 4:
                        # Random policy
                        row_str += "  *  "
                    else:
                        # Show arrows for best actions
                        symbols = ""
                        for a in best_actions:
                            symbols += self.action_symbols[a]
                        row_str += f"{symbols:^5}"
                        
                row_str += " | " if col < self.size - 1 else ""
            print(row_str)
            
            if row < self.size - 1:
                print("-" * 50)