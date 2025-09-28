"""
Stochastic Windy Gridworld Implementation
Author: Mehar
Course: NEU Self-Improving AI
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import random


class WindyGridworld:
    """
    Implementation of Windy Gridworld with stochastic wind
    Based on Sutton & Barto description
    """
    
    def __init__(self, height: int = 7, width: int = 10, stochastic: bool = True):
        """
        Initialize Windy Gridworld
        
        Args:
            height: Grid height
            width: Grid width  
            stochastic: Whether to use stochastic wind
        """
        self.height = height
        self.width = width
        self.stochastic = stochastic
        
        # Define start and goal positions
        self.start_pos = (3, 0)  # Middle left
        self.goal_pos = (3, 7)   # Middle right area
        
        # Wind strength by column
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right  
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        self.action_names = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN', 
            3: 'LEFT'
        }
        
        self.n_actions = len(self.actions)
        
        # Initialize episode tracking
        self.reset()
        
    def reset(self) -> Tuple[int, int]:
        """
        Reset environment to start state
        
        Returns:
            Starting position
        """
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.current_pos
    
    def position_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index"""
        row, col = pos
        return row * self.width + col
    
    def state_to_position(self, state: int) -> Tuple[int, int]:
        """Convert state index to position"""
        row = state // self.width
        col = state % self.width
        return (row, col)
    
    def is_terminal(self, pos: Tuple[int, int]) -> bool:
        """Check if position is goal"""
        return pos == self.goal_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take action in environment
        
        Args:
            action: Action index
            
        Returns:
            (next_position, reward, done)
        """
        if self.is_terminal(self.current_pos):
            return self.current_pos, 0.0, True
            
        # Get base movement from action
        row, col = self.current_pos
        delta_row, delta_col = self.actions[action]
        
        # Apply action
        new_row = row + delta_row
        new_col = col + delta_col
        
        # Apply wind effect
        base_wind = self.wind_strength[col] if col < len(self.wind_strength) else 0
        
        if self.stochastic and base_wind > 0:
            # Stochastic wind: 80% base, 10% base+1, 10% no wind
            rand = random.random()
            if rand < 0.1:
                wind_effect = 0
            elif rand < 0.9:
                wind_effect = base_wind
            else:
                wind_effect = base_wind + 1
        else:
            wind_effect = base_wind
            
        # Wind pushes up (negative row direction)
        new_row -= wind_effect
        
        # Clip to boundaries
        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))
        
        self.current_pos = (new_row, new_col)
        self.steps_taken += 1
        
        # Calculate reward
        if self.is_terminal(self.current_pos):
            reward = 0.0
            done = True
        else:
            reward = -1.0
            done = False
            
        return self.current_pos, reward, done
    
    def get_transition_probabilities(self, state: Tuple[int, int], action: int) -> Dict[Tuple[int, int], float]:
        """
        Get all possible next states and their probabilities
        
        Args:
            state: Current position
            action: Action to take
            
        Returns:
            Dictionary of {next_state: probability}
        """
        if self.is_terminal(state):
            return {state: 1.0}
            
        row, col = state
        delta_row, delta_col = self.actions[action]
        
        # Calculate new position after action
        new_row = row + delta_row
        new_col = col + delta_col
        
        # Get base wind strength
        base_wind = self.wind_strength[col] if col < len(self.wind_strength) else 0
        
        transitions = {}
        
        if not self.stochastic or base_wind == 0:
            # Deterministic case or no wind
            final_row = new_row - base_wind
            final_row = max(0, min(self.height - 1, final_row))
            final_col = max(0, min(self.width - 1, new_col))
            transitions[(final_row, final_col)] = 1.0
        else:
            # Stochastic wind
            for wind_effect, prob in [(0, 0.1), (base_wind, 0.8), (base_wind + 1, 0.1)]:
                final_row = new_row - wind_effect
                final_row = max(0, min(self.height - 1, final_row))
                final_col = max(0, min(self.width - 1, new_col))
                
                next_pos = (final_row, final_col)
                if next_pos in transitions:
                    transitions[next_pos] += prob
                else:
                    transitions[next_pos] = prob
                    
        return transitions
    
    def render(self, value_function: Optional[np.ndarray] = None, 
               policy: Optional[np.ndarray] = None):
        """
        Display the gridworld
        
        Args:
            value_function: Optional value function to display
            policy: Optional policy to display
        """
        print("\n" + "=" * 60)
        print("Windy Gridworld")
        print("=" * 60)
        
        # Display wind strengths
        print("\nWind strength by column:")
        print("Col: ", end="")
        for c in range(self.width):
            print(f"{c:3}", end="")
        print("\nWind:", end="")
        for c in range(self.width):
            wind = self.wind_strength[c] if c < len(self.wind_strength) else 0
            print(f"{wind:3}", end="")
        print("\n")
        
        # Display grid
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                
                # Determine what to display
                if pos == self.current_pos and pos == self.goal_pos:
                    symbol = "GC"  # Goal reached
                elif pos == self.current_pos:
                    symbol = "C "  # Current position
                elif pos == self.goal_pos:
                    symbol = "G "  # Goal
                elif pos == self.start_pos:
                    symbol = "S "  # Start
                else:
                    symbol = ". "  # Empty
                    
                print(symbol, end=" ")
            print()
            
        print("\nLegend: S=Start, G=Goal, C=Current, .=Empty")
        print(f"Steps taken: {self.steps_taken}")
        
        # Display value function if provided
        if value_function is not None:
            print("\nValue Function:")
            for row in range(self.height):
                for col in range(self.width):
                    state_idx = self.position_to_state((row, col))
                    print(f"{value_function[state_idx]:6.2f}", end=" ")
                print()
                
        # Display policy if provided  
        if policy is not None:
            print("\nPolicy:")
            symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
            for row in range(self.height):
                for col in range(self.width):
                    if (row, col) == self.goal_pos:
                        print("  G  ", end=" ")
                    else:
                        state_idx = self.position_to_state((row, col))
                        best_action = np.argmax(policy[state_idx])
                        print(f"  {symbols[best_action]}  ", end=" ")
                print()