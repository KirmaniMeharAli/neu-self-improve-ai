"""
Value Model for PAG with A*-PO

"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict


class ValueModel(nn.Module):
    """Value network for state evaluation."""

    def __init__(self, model_name: str, hidden_size: int = 768, device: torch.device = None):
        super().__init__()

        # Automatically select GPU if available
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Base encoder (use float16 only if GPU supports it)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)

        # Value head layers
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        ).to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute state values."""

        # Move inputs to the same device
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        # Encode text inputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pool hidden states (average pooling)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)

        # Match dtype between encoder and head
        pooled = pooled.to(next(self.value_head.parameters()).dtype)

        # Compute scalar value
        value = self.value_head(pooled)
        return value.squeeze(-1)

    def compute_advantages(self,
                           rewards: torch.Tensor,
                           values: torch.Tensor,
                           next_values: torch.Tensor,
                           dones: torch.Tensor,
                           gamma: float = 0.99) -> torch.Tensor:
        """Compute normalized advantage estimates."""

        # Move all tensors to same device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        next_values = next_values.to(self.device)
        dones = dones.to(self.device)

        # Compute returns
        returns = rewards + gamma * next_values * (1 - dones)

        # Calculate advantages
        advantages = returns - values

        # Normalize to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
