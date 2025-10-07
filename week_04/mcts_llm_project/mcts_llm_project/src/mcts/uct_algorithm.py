"""
MCTS-UCT Algorithm Implementation
Author: Mehar
Course: NEU Self-Improving AI
Date: 2025

Implements the UCT (Upper Confidence Bound for Trees) algorithm
combining MCTS with UCB1 selection strategy.
"""

import time
import random
import numpy as np
from typing import Any, List, Optional, Callable, Dict, Tuple
from dataclasses import dataclass
import logging

from .node import MCTSNode


@dataclass
class UCTConfig:
    """Configuration for UCT algorithm."""
    num_simulations: int = 1000
    exploration_constant: float = 1.414  # sqrt(2)
    simulation_depth_limit: int = 100
    time_limit: Optional[float] = None
    verbose: bool = False
    seed: Optional[int] = None


class UCTAlgorithm:
    """
    Implements the UCT (Upper Confidence Bound for Trees) algorithm.
    
    This combines Monte Carlo Tree Search with UCB1 for node selection,
    providing a principled exploration-exploitation tradeoff.
    """
    
    def __init__(
        self,
        config: UCTConfig,
        get_actions: Callable[[Any], List[Any]],
        take_action: Callable[[Any, Any], Tuple[Any, float, bool]],
        evaluate_state: Optional[Callable[[Any], float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize UCT algorithm.
        
        Args:
            config: Algorithm configuration
            get_actions: Function to get valid actions from a state
            take_action: Function to take action and get (next_state, reward, done)
            evaluate_state: Optional function for state evaluation
            logger: Optional logger for debugging
        """
        self.config = config
        self.get_actions = get_actions
        self.take_action = take_action
        self.evaluate_state = evaluate_state or self._default_rollout
        self.logger = logger or self._setup_default_logger()
        
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            
        self.statistics = {
            'total_simulations': 0,
            'average_depth': 0,
            'max_tree_size': 0
        }
    
    def search(self, initial_state: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform MCTS-UCT search from the given initial state.
        
        Args:
            initial_state: Starting state for search
            
        Returns:
            Tuple of (best_action, search_statistics)
        """
        # Initialize root node
        root = MCTSNode(
            state=initial_state,
            untried_actions=self.get_actions(initial_state)
        )
        
        start_time = time.time()
        simulation_count = 0
        
        # Main MCTS loop
        while not self._should_stop(simulation_count, start_time):
            # Phase 1: Selection
            node = self._selection(root)
            
            # Phase 2: Expansion
            if not node.is_terminal and node.is_fully_expanded():
                node = self._expansion(node)
            
            # Phase 3: Simulation (Rollout)
            reward = self._simulation(node)
            
            # Phase 4: Backpropagation
            self._backpropagation(node, reward)
            
            simulation_count += 1
            
            if self.config.verbose and simulation_count % 100 == 0:
                self.logger.info(f"Completed {simulation_count} simulations")
        
        # Select best action based on visit count
        best_child = root.most_visited_child()
        best_action = best_child.action if best_child else None
        
        # Gather statistics
        stats = self._gather_statistics(root, simulation_count, time.time() - start_time)
        
        return best_action, stats
    
    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Navigate down the tree using UCB1.
        
        Args:
            node: Starting node for selection
            
        Returns:
            Selected node for expansion or simulation
        """
        current = node
        
        while not current.is_terminal:
            if not current.is_fully_expanded():
                return current
            else:
                current = current.best_child(self.config.exploration_constant)
                if current is None:
                    break
                    
        return current
    
    def _expansion(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add a new child to the tree.
        
        Args:
            node: Node to expand
            
        Returns:
            Newly created child node
        """
        if not node.untried_actions:
            return node
            
        # Select random untried action
        action = random.choice(node.untried_actions)
        
        # Take action to get next state
        next_state, _, done = self.take_action(node.state, action)
        
        # Create child node
        child = node.add_child(next_state, action)
        child.is_terminal = done
        
        if not done:
            child.untried_actions = self.get_actions(next_state)
            
        return child
    
    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: Play out randomly or use evaluation function.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Reward value from simulation
        """
        if node.is_terminal:
            return self.evaluate_state(node.state)
            
        # Use provided evaluation function or default rollout
        return self.evaluate_state(node.state)
    
    def _default_rollout(self, state: Any) -> float:
        """
        Default rollout policy using random actions.
        
        Args:
            state: Starting state for rollout
            
        Returns:
            Final reward from rollout
        """
        current_state = state
        total_reward = 0
        depth = 0
        
        while depth < self.config.simulation_depth_limit:
            actions = self.get_actions(current_state)
            if not actions:
                break
                
            action = random.choice(actions)
            next_state, reward, done = self.take_action(current_state, action)
            total_reward += reward
            
            if done:
                break
                
            current_state = next_state
            depth += 1
            
        return total_reward
    
    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update statistics up the tree.
        
        Args:
            node: Leaf node to start backpropagation
            reward: Reward value to propagate
        """
        current = node
        
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def _should_stop(self, simulation_count: int, start_time: float) -> bool:
        """
        Check if search should stop based on constraints.
        
        Args:
            simulation_count: Current number of simulations
            start_time: Start time of search
            
        Returns:
            True if search should stop
        """
        # Check simulation limit
        if simulation_count >= self.config.num_simulations:
            return True
            
        # Check time limit if specified
        if self.config.time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed >= self.config.time_limit:
                return True
                
        return False
    
    def _gather_statistics(
        self, 
        root: MCTSNode, 
        simulations: int, 
        elapsed_time: float
    ) -> Dict[str, Any]:
        """
        Gather search statistics for analysis.
        
        Args:
            root: Root node of search tree
            simulations: Total simulations performed
            elapsed_time: Time taken for search
            
        Returns:
            Dictionary of statistics
        """
        tree_size = self._count_nodes(root)
        avg_depth = self._calculate_average_depth(root)
        
        return {
            'simulations': simulations,
            'elapsed_time': elapsed_time,
            'tree_size': tree_size,
            'average_depth': avg_depth,
            'root_visits': root.visits,
            'best_action_visits': root.most_visited_child().visits if root.children else 0,
            'simulations_per_second': simulations / elapsed_time if elapsed_time > 0 else 0
        }
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _calculate_average_depth(self, node: MCTSNode, depth: int = 0) -> float:
        """Calculate average depth of tree."""
        if not node.children:
            return depth
            
        total_depth = 0
        for child in node.children:
            total_depth += self._calculate_average_depth(child, depth + 1)
            
        return total_depth / len(node.children) if node.children else depth
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger if none provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
