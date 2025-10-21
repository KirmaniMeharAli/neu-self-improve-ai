"""
A*-PO (A-Star Policy Optimization) Algorithm - GPU Optimized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class AStarPO:
    """A*-PO optimizer for policy improvement with GPU support."""
    
    def __init__(self,
                 policy_model,
                 value_model,
                 learning_rate: float = 1e-5,
                 advantage_weight: float = 1.0,
                 kl_weight: float = 0.1,
                 entropy_weight: float = 0.01,
                 clip_range: float = 0.2,
                 max_grad_norm: float = 0.5):
        """
        Initialize A*-PO optimizer.
        """
        self.policy_model = policy_model
        self.value_model = value_model

        # ✅ Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        
        self.advantage_weight = advantage_weight
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        
        # ✅ AdamW Optimizers (GPU compatible)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=learning_rate * 2
        )
    
    # -------------------------------------------------------------------------
    # ADVANTAGE ESTIMATION (A* Search)
    # -------------------------------------------------------------------------
    def compute_optimal_advantage(self,
                                 states: torch.Tensor,
                                 actions: torch.Tensor,
                                 rewards: torch.Tensor,
                                 next_states: torch.Tensor,
                                 dones: torch.Tensor) -> torch.Tensor:
        """Compute optimal advantage using A* heuristic."""
        batch_size = states.shape[0]
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
        )
        
        with torch.no_grad():
            current_values = self.value_model(
                states[:, :, 0],
                states[:, :, 1]
            )
            next_values = self.value_model(
                next_states[:, :, 0],
                next_states[:, :, 1]
            )
            
            heuristic_values = rewards + 0.99 * next_values * (1 - dones)
            advantages = torch.zeros_like(rewards, device=self.device)
            
            for i in range(batch_size):
                best_value = self._astar_search(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    current_values[i],
                    next_values[i]
                )
                advantages[i] = best_value - current_values[i]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    # -------------------------------------------------------------------------
    # A* SEARCH CORE
    # -------------------------------------------------------------------------
    def _astar_search(self,
                     state: torch.Tensor,
                     action: torch.Tensor,
                     reward: float,
                     next_state: torch.Tensor,
                     current_value: float,
                     next_value: float,
                     search_depth: int = 3) -> float:
        """Simplified A* search for local value refinement."""
        best_value = reward + 0.99 * next_value
        
        for _ in range(search_depth):
            with torch.no_grad():
                logits = self.policy_model(
                    state.unsqueeze(0).to(self.device),
                    torch.ones_like(state[0]).unsqueeze(0).to(self.device)
                )['logits']
                
                probs = F.softmax(logits[0, -1], dim=-1)
                alt_actions = torch.multinomial(probs, 3, replacement=False)
                
                for alt_action in alt_actions:
                    alt_value = self._estimate_action_value(
                        state, alt_action, next_state
                    )
                    best_value = max(best_value, alt_value)
        
        return best_value
    
    def _estimate_action_value(self,
                              state: torch.Tensor,
                              action: int,
                              next_state: torch.Tensor) -> float:
        """Estimate value of taking specific action."""
        with torch.no_grad():
            value = self.value_model(
                next_state.unsqueeze(0).to(self.device),
                torch.ones_like(next_state[0]).unsqueeze(0).to(self.device)
            )
        return value.item()
    
    # -------------------------------------------------------------------------
    # MAIN TRAINING UPDATE
    # -------------------------------------------------------------------------
    def update(self, batch: Dict) -> Dict:
        """Update policy and value networks based on batch trajectories."""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute A*-based advantage
        advantages = self.compute_optimal_advantage(
            states, actions, rewards, next_states, dones
        )
        
        # Policy update
        policy_loss, policy_stats = self._update_policy(
            states, actions, advantages, old_log_probs
        )
        
        # Value update
        value_loss = self._update_value(
            states, rewards, next_states, dones
        )
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            **policy_stats
        }
    
    # -------------------------------------------------------------------------
    # POLICY UPDATE
    # -------------------------------------------------------------------------
    def _update_policy(self,
                      states: torch.Tensor,
                      actions: torch.Tensor,
                      advantages: torch.Tensor,
                      old_log_probs: torch.Tensor) -> Tuple[float, Dict]:
        """Update policy network parameters."""
        log_probs = self.policy_model.compute_log_probs(
            states[:, :, 0],
            states[:, :, 1],
            actions
        )
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL + entropy regularization
        kl_div = self.policy_model.compute_kl_divergence(
            states[:, :, 0],
            states[:, :, 1]
        )
        policy_loss += self.kl_weight * kl_div
        
        entropy = -(log_probs * torch.exp(log_probs)).mean()
        policy_loss -= self.entropy_weight * entropy
        
        # Optimize
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        return policy_loss.item(), {
            'kl_div': kl_div.item(),
            'entropy': entropy.item(),
            'ratio': ratio.mean().item()
        }
    
    # -------------------------------------------------------------------------
    # VALUE UPDATE
    # -------------------------------------------------------------------------
    def _update_value(self,
                     states: torch.Tensor,
                     rewards: torch.Tensor,
                     next_states: torch.Tensor,
                     dones: torch.Tensor) -> float:
        """Update value network."""
        values = self.value_model(
            states[:, :, 0],
            states[:, :, 1]
        )
        
        with torch.no_grad():
            next_values = self.value_model(
                next_states[:, :, 0],
                next_states[:, :, 1]
            )
            returns = rewards + 0.99 * next_values * (1 - dones)
        
        value_loss = F.mse_loss(values, returns)
        
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()
