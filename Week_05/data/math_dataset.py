"""
MATH Dataset Handler for PAG with A*-PO
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import re
from typing import Dict, List, Tuple, Optional
import torch

class MATHDataset(Dataset):
    """MATH competition dataset handler."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 2048, 
                 subset_size: Optional[int] = None, device: str = 'cpu'):
        """
        Initialize MATH dataset.

        Args:
            csv_path: Path to the CSV file
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            subset_size: If provided, use only this many samples
            device: Torch device to move tensors to
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        # Load data
        self.data = pd.read_csv(csv_path)

        # Handle subset
        if subset_size and subset_size < len(self.data):
            self.data = self.data.sample(n=subset_size, random_state=42)

        # Preprocess
        self.problems = self.data['problem'].tolist()
        self.solutions = self.data['solution'].tolist()
        self.levels = self.data['level'].tolist() if 'level' in self.data else None
        self.types = self.data['type'].tolist() if 'type' in self.data else None

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from dataset."""
        problem = self.problems[idx]
        solution = self.solutions[idx]

        # Format prompt
        prompt = self._format_prompt(problem)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Move tensors to the correct device
        input_ids = inputs['input_ids'].squeeze().to(self.device)
        attention_mask = inputs['attention_mask'].squeeze().to(self.device)

        # Extract final answer from solution
        final_answer = self._extract_final_answer(solution)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'problem': problem,
            'solution': solution,
            'final_answer': final_answer,
            'level': self.levels[idx] if self.levels else None,
            'type': self.types[idx] if self.types else None
        }

    def _format_prompt(self, problem: str) -> str:
        """Format problem into prompt."""
        return (
            "Solve the following math problem step by step.\n\n"
            f"Problem: {problem}\n\n"
            "Solution: Let me solve this step by step.\n"
        )

    def _extract_final_answer(self, solution: str) -> str:
        """Extract final answer from solution."""
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            return boxed_match.group(1)

        # Look for final numeric answer
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        if numbers:
            return numbers[-1]

        return solution.strip().split('\n')[-1]

def create_dataloaders(config: dict, tokenizer, device: str = 'cpu'):
    """Create training and evaluation dataloaders."""

    # Training dataset (full)
    train_dataset = MATHDataset(
        csv_path=config['dataset']['train_path'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        device=device
    )

    # Evaluation dataset (subset)
    eval_dataset = MATHDataset(
        csv_path=config['dataset']['train_path'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        subset_size=config['dataset']['eval_subset_size'],
        device=device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0  # Fixed for Windows
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, eval_loader
