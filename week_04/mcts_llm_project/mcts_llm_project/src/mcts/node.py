"""
MCTS Node Implementation
Author: Mehar
Course: NEU Self-Improving AI
Date: 2025

This module implements the Node class for Monte Carlo Tree Search.
Following best practices from Sutton & Barto Ch. 13.
"""

import math
import numpy as np
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """
    Represents a node in the MCTS tree structure.
    
    Attributes:
        state: Current state representation
        parent: Parent node reference
        action: Action that led to this node
        children: List of child nodes
        visits: Number of times this node has been visited
        value: Cumulative value from all simulations
        untried_actions: Actions not yet explored from this state
        is_terminal: Whether this is a terminal state
    """
    
    state: Any
    parent: Optional['MCTSNode'] = None
    action: Optional[Any] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Any] = field(default_factory=list)
    is_terminal: bool = False
    
    def __post_init__(self):
        """Initialize node properties after creation."""
        self._results = {}
        self._untried_actions_set = set(self.untried_actions)
        
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried from this node.
        
        Returns:
            bool: True if all actions have been explored
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param: float = 1.414) -> Optional['MCTSNode']:
        """
        Select best child using UCB1 formula.
        
        UCB1 = vi + c * sqrt(ln(N) / ni)
        where:
            vi = average value of child i
            c = exploration constant
            N = parent visit count
            ni = child visit count
            
        Args:
            c_param: Exploration constant (default sqrt(2))
            
        Returns:
            Best child node according to UCB1
        """
        if not self.children:
            return None
            
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                # Unvisited nodes get maximum priority
                ucb1 = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                ucb1 = exploitation + exploration
            choices_weights.append(ucb1)
            
        best_index = np.argmax(choices_weights)
        return self.children[best_index]
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """
        Return the child with the highest visit count.
        Used for final move selection.
        
        Returns:
            Most visited child node
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def add_child(self, state: Any, action: Any) -> 'MCTSNode':
        """
        Add a new child node for the given state and action.
        
        Args:
            state: State of the new child
            action: Action that leads to this child
            
        Returns:
            The newly created child node
        """
        child = MCTSNode(
            state=state,
            parent=self,
            action=action,
            untried_actions=[]  # Will be set by environment
        )
        self.children.append(child)
        
        # Remove action from untried actions
        if action in self.untried_actions:
            self.untried_actions.remove(action)
            
        return child
    
    def update(self, result: float) -> None:
        """
        Update node statistics after a simulation.
        
        Args:
            result: Value obtained from simulation
        """
        self.visits += 1
        self.value += result
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get node statistics for analysis.
        
        Returns:
            Dictionary containing node statistics
        """
        avg_value = self.value / self.visits if self.visits > 0 else 0
        return {
            'visits': self.visits,
            'total_value': self.value,
            'average_value': avg_value,
            'num_children': len(self.children),
            'is_terminal': self.is_terminal,
            'is_fully_expanded': self.is_fully_expanded()
        }
    
    def tree_to_string(self, indent: int = 0) -> str:
        """
        Generate string representation of tree for debugging.
        
        Args:
            indent: Current indentation level
            
        Returns:
            String representation of tree structure
        """
        s = self._indent_string(indent) + str(self)
        for child in self.children:
            s += child.tree_to_string(indent + 1)
        return s
    
    def _indent_string(self, indent: int) -> str:
        """Helper method for tree string generation."""
        return '\n' + '  ' * indent
    
    def __str__(self) -> str:
        """String representation of node."""
        avg_value = self.value / self.visits if self.visits > 0 else 0
        return f"[A:{self.action} V:{self.visits} Q:{avg_value:.3f}]"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"MCTSNode(action={self.action}, visits={self.visits}, "
                f"value={self.value:.3f}, children={len(self.children)})")
