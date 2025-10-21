"""
Training loop for PAG with A*-PO - GPU Version with Step Limiting
"""
import torch
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import json
import os

class Trainer:
    """Trainer for PAG with A*-PO."""
    
    def __init__(self,
                 pag_system,
                 train_loader,
                 eval_loader,
                 config: Dict,
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize trainer.
        
        Args:
            pag_system: PAG system with policy and value models
            train_loader: Training dataloader
            eval_loader: Evaluation dataloader  
            config: Training configuration
            device: Device to use (cuda/cpu)
            checkpoint_dir: Directory for checkpoints
        """
        self.pag = pag_system
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb if configured
        if config['logging'].get('use_wandb', False):
            try:
                wandb.init(
                    project=config['logging']['project_name'],
                    config=config,
                    mode='offline'  # Use offline mode
                )
            except:
                print("W&B initialization failed, continuing without logging")
    
    def train(self):
        """Main training loop."""
        global_step = 0
        
        # Enable mixed precision training on GPU
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Training
            train_stats = self._train_epoch(epoch, global_step, scaler)
            global_step += min(len(self.train_loader), self.config['training'].get('steps_per_epoch', len(self.train_loader)))
            
            # Evaluation
            eval_frequency = self.config['training'].get('eval_frequency', 100)
            if (epoch + 1) % eval_frequency == 0 or epoch == self.config['training']['num_epochs'] - 1:
                eval_stats = self._evaluate()
                print(f"Evaluation - Accuracy: {eval_stats['accuracy']:.4f}")
                
                if self.config['logging'].get('use_wandb', False):
                    try:
                        wandb.log({**train_stats, **eval_stats}, step=global_step)
                    except:
                        pass
            
            # Save checkpoint
            save_frequency = self.config['training'].get('save_frequency', 100)
            if (epoch + 1) % save_frequency == 0 or epoch == self.config['training']['num_epochs'] - 1:
                self._save_checkpoint(epoch, global_step)
    
    def _train_epoch(self, epoch: int, global_step: int, scaler=None) -> Dict:
        """Train for one epoch."""
        epoch_stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'rewards': [],
            'correct': 0,
            'total': 0
        }
        
        # Get max steps for this epoch
        max_steps = self.config['training'].get('steps_per_epoch', len(self.train_loader))
        
        # Create progress bar with correct total
        actual_steps = min(max_steps, len(self.train_loader))
        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=actual_steps,
            desc=f"Training Epoch {epoch + 1}"
        )
        
        for batch_idx, batch in progress_bar:
            # STOP AFTER CONFIGURED STEPS
            if batch_idx >= max_steps:
                print(f"\nCompleted {batch_idx} training steps (configured limit)")
                break
            
            # Move batch to GPU
            if self.device == 'cuda':
                batch = {k: v.cuda() if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            
            # Generate solutions with MCTS
            problems = batch['problem']
            solutions = []
            
            # Use mixed precision on GPU
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                for problem in problems:
                    solution_dict = self.pag.generate_solution(problem)
                    solutions.append(solution_dict['solution'])
            
            # Self-improvement step
            update_stats = self.pag.self_improve(problems, solutions)
            
            if update_stats:
                epoch_stats['policy_loss'] += update_stats.get('policy_loss', 0)
                epoch_stats['value_loss'] += update_stats.get('value_loss', 0)
            
            # Track accuracy
            for i, (problem, solution) in enumerate(zip(problems, solutions)):
                is_correct = self._check_answer(
                    solution,
                    batch['final_answer'][i]
                )
                epoch_stats['correct'] += int(is_correct)
                epoch_stats['total'] += 1
            
            # Update progress bar
            if epoch_stats['total'] > 0:
                accuracy = epoch_stats['correct'] / epoch_stats['total']
                progress_bar.set_postfix({
                    'acc': f"{accuracy:.3f}",
                    'p_loss': f"{epoch_stats['policy_loss']/(batch_idx+1):.4f}",
                    'v_loss': f"{epoch_stats['value_loss']/(batch_idx+1):.4f}",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if self.device == 'cuda' else 'N/A'
                })
            
            # Clear GPU cache periodically
            if self.device == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Log periodically
            if (batch_idx + 1) % self.config['logging']['log_frequency'] == 0:
                if self.config['logging'].get('use_wandb', False):
                    try:
                        wandb.log({
                            'train_accuracy': accuracy,
                            'policy_loss': epoch_stats['policy_loss'] / (batch_idx + 1),
                            'value_loss': epoch_stats['value_loss'] / (batch_idx + 1),
                            'gpu_memory_gb': torch.cuda.memory_allocated()/1e9 if self.device == 'cuda' else 0
                        }, step=global_step + batch_idx)
                    except:
                        pass
        
        # Close progress bar
        progress_bar.close()
        
        # Compute epoch statistics
        num_batches = min(max_steps, len(self.train_loader))
        epoch_stats['accuracy'] = epoch_stats['correct'] / max(epoch_stats['total'], 1)
        epoch_stats['policy_loss'] = epoch_stats['policy_loss'] / max(num_batches, 1)
        epoch_stats['value_loss'] = epoch_stats['value_loss'] / max(num_batches, 1)
        
        print(f"\nEpoch completed: {num_batches} steps, Accuracy: {epoch_stats['accuracy']:.4f}")
        
        return epoch_stats
    
    def _evaluate(self) -> Dict:
        """Evaluate on test set."""
        eval_stats = {
            'accuracy': 0,
            'correct': 0,
            'total': 0
        }
        
        # Limit evaluation steps for speed
        max_eval_batches = 5  # Only evaluate on 5 batches for quick testing
        
        progress_bar = tqdm(
            enumerate(self.eval_loader), 
            total=min(max_eval_batches, len(self.eval_loader)),
            desc="Evaluating"
        )
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                for batch_idx, batch in progress_bar:
                    # Stop after max_eval_batches
                    if batch_idx >= max_eval_batches:
                        break
                    
                    # Move to GPU
                    if self.device == 'cuda':
                        batch = {k: v.cuda() if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                    
                    problems = batch['problem']
                    
                    for i, problem in enumerate(problems):
                        # Generate solution
                        solution_dict = self.pag.generate_solution(problem)
                        solution = solution_dict['solution']
                        
                        # Check answer
                        is_correct = self._check_answer(
                            solution,
                            batch['final_answer'][i]
                        )
                        
                        eval_stats['correct'] += int(is_correct)
                        eval_stats['total'] += 1
                    
                    # Update progress
                    accuracy = eval_stats['correct'] / max(eval_stats['total'], 1)
                    progress_bar.set_postfix({
                        'accuracy': f"{accuracy:.3f}",
                        'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if self.device == 'cuda' else 'N/A'
                    })
        
        progress_bar.close()
        eval_stats['accuracy'] = eval_stats['correct'] / max(eval_stats['total'], 1)
        
        return eval_stats
    
    def _check_answer(self, solution: str, target_answer: str) -> bool:
        """Check if solution contains correct answer."""
        # Extract answer from solution
        import re
        
        # Look for boxed answer in solution
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            pred_answer = boxed_match.group(1).strip()
        else:
            # Try to find numeric answer
            numbers = re.findall(r'-?\d+\.?\d*', solution)
            pred_answer = numbers[-1] if numbers else ""
        
        # Normalize and compare
        pred_answer = pred_answer.replace(" ", "").lower()
        target_answer = str(target_answer).replace(" ", "").lower()
        
        return pred_answer == target_answer
    
    def _save_checkpoint(self, epoch: int, global_step: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'config': self.config
        }
        
        # Only save state dicts if models are accessible
        try:
            checkpoint['policy_state'] = self.pag.policy_model.state_dict()
            checkpoint['value_state'] = self.pag.value_model.state_dict()
        except:
            print("Warning: Could not save model states")
        
        path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch{epoch}_step{global_step}.pt'
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")