"""
Monte Carlo Tree Search for PAG - GPU Version
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass
import math

@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    state: str
    parent: Optional['MCTSNode'] = None
    action: Optional[str] = None
    children: Dict[str, 'MCTSNode'] = None
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Average value of node."""
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    @property
    def is_leaf(self) -> bool:
        """Check if node is leaf."""
        return len(self.children) == 0
    
    def ucb_score(self, c_puct: float = 1.4) -> float:
        """Calculate UCB score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value + exploration

class MCTS:
    """Monte Carlo Tree Search for solution generation."""
    
    def __init__(self,
                 policy_model,
                 value_model,
                 c_puct: float = 1.4,
                 num_simulations: int = 50,
                 max_depth: int = 10):
        """
        Initialize MCTS.
        
        Args:
            policy_model: Policy network for action probabilities
            value_model: Value network for state evaluation
            c_puct: Exploration constant
            num_simulations: Number of MCTS simulations
            max_depth: Maximum search depth
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.device = policy_model.device  # Use policy model's device
    
    def search(self, initial_state: str) -> Dict:
        """
        Run MCTS from initial state.
        
        Args:
            initial_state: Starting problem state
            
        Returns:
            Dictionary with best action and statistics
        """
        root = MCTSNode(state=initial_state)
        
        # Use GPU batching for efficiency
        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]
                depth = 0
                
                # Selection phase
                while not node.is_leaf and depth < self.max_depth:
                    node = self._select_child(node)
                    search_path.append(node)
                    depth += 1
                
                # Expansion phase
                if depth < self.max_depth and not self._is_terminal(node.state):
                    self._expand(node)
                    if node.children:
                        node = self._select_child(node)
                        search_path.append(node)
                
                # Evaluation phase
                value = self._evaluate(node.state)
                
                # Backup phase
                self._backup(search_path, value)
        
        # Select best action
        if root.children:
            best_child = max(root.children.values(), key=lambda n: n.visits)
            return {
                'action': best_child.action,
                'value': best_child.value,
                'visits': best_child.visits,
                'root_value': root.value,
                'total_visits': root.visits
            }
        else:
            return {
                'action': '',
                'value': 0.0,
                'visits': 0,
                'root_value': 0.0,
                'total_visits': 0
            }
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        return max(node.children.values(), 
                  key=lambda n: n.ucb_score(self.c_puct))
    
    def _expand(self, node: MCTSNode):
        """Expand node by generating possible actions."""
        # Get action probabilities from policy
        actions, priors = self._get_action_probs(node.state)
        
        # Create child nodes
        for action, prior in zip(actions, priors):
            if action not in node.children:
                new_state = self._apply_action(node.state, action)
                child = MCTSNode(
                    state=new_state,
                    parent=node,
                    action=action,
                    prior=prior
                )
                node.children[action] = child
    
    def _evaluate(self, state: str) -> float:
        """Evaluate state using value network."""
        # Tokenize state
        inputs = self.policy_model.tokenizer(
            state,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512  # Limit length for GPU memory
        )
        
        # Move to GPU
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get value estimate
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                value = self.value_model(input_ids, attention_mask)
        
        return value.item() if value.numel() == 1 else value[0].item()
    
    def _backup(self, search_path: List[MCTSNode], value: float):
        """Backup value through search path."""
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = -value  # Alternate perspective
    
    def _get_action_probs(self, state: str) -> Tuple[List[str], List[float]]:
        """Get action probabilities from policy."""
        # Tokenize
        inputs = self.policy_model.tokenizer(
            state,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to GPU
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate next tokens
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                outputs = self.policy_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,  # Reduced for speed
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    temperature=0.8
                )
        
        # Get top actions and probabilities
        if outputs.scores:
            scores = outputs.scores[0][0]  # First token scores
            probs = torch.softmax(scores, dim=-1)
            top_k = 5
            top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(-1)))
            
            # Decode actions
            actions = [
                self.policy_model.tokenizer.decode(idx)
                for idx in top_ids
            ]
            
            return actions, top_probs.cpu().numpy().tolist()
        else:
            return [''], [1.0]
    
    def _apply_action(self, state: str, action: str) -> str:
        """Apply action to state to get new state."""
        return state + " " + action
    
    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal."""
        # Check for solution indicators
        terminal_indicators = ['Therefore', 'The answer is', '\\boxed{', 'Final answer:']
        return any(indicator in state for indicator in terminal_indicators)
