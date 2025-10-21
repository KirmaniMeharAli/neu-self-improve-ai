"""
Policy Model for PAG with A*-PO - GPU Version
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

class PolicyModel(nn.Module):
    """Policy model wrapper for LLM."""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model and tokenizer with GPU optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        # Move to GPU if available
        if self.device == 'cuda':
            self.model = self.model.cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reference model for KL divergence
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        if self.device == 'cuda':
            self.ref_model = self.ref_model.cuda()
        self.ref_model.eval()
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        """Forward pass through policy model."""
        
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }
    
    def generate_trajectory(self, 
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           max_new_tokens: int = 512,
                           temperature: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate solution trajectory."""
        
        # Move to GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generated_ids = outputs.sequences
        scores = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
        
        return generated_ids, scores
    
    def compute_log_probs(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         action_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for actions."""
        
        # Ensure GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        action_ids = action_ids.to(self.device)
        
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits']
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual actions
        action_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=action_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return action_log_probs
    
    def compute_kl_divergence(self,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from reference model."""
        
        # Ensure GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Current policy logits
        curr_outputs = self.forward(input_ids, attention_mask)
        curr_logits = curr_outputs['logits']
        
        # Reference policy logits
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            ref_logits = ref_outputs.logits
        
        # Compute KL
        curr_probs = F.softmax(curr_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl_div = torch.sum(
            curr_probs * (torch.log(curr_probs + 1e-10) - torch.log(ref_probs + 1e-10)),
            dim=-1
        )
        
        return kl_div.mean()