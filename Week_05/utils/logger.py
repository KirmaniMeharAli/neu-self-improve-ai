"""
Logging utilities for PAG with A*-PO
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
import wandb
from pathlib import Path

class Logger:
    """Custom logger for training and evaluation."""
    
    def __init__(self,
                 name: str = "PAG-ASTAR",
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 use_wandb: bool = True,
                 wandb_project: Optional[str] = None,
                 wandb_config: Optional[Dict] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_config: W&B configuration
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logger(log_level)
        
        # Setup console logging
        self.setup_console_logger(log_level)
        
        # Setup wandb
        self.use_wandb = use_wandb
        if use_wandb and wandb_project:
            self.setup_wandb(wandb_project, wandb_config)
        
        # Metrics storage
        self.metrics_history = []
    
    def setup_file_logger(self, log_level: str):
        """Setup file logger."""
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Create formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Get logger and add handler
        self.file_logger = logging.getLogger(f"{self.name}_file")
        self.file_logger.setLevel(getattr(logging, log_level))
        self.file_logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def setup_console_logger(self, log_level: str):
        """Setup console logger."""
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Create formatter (simpler for console)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Get logger and add handler
        self.console_logger = logging.getLogger(f"{self.name}_console")
        self.console_logger.setLevel(getattr(logging, log_level))
        self.console_logger.addHandler(console_handler)
    
    def setup_wandb(self, project: str, config: Optional[Dict] = None):
        """Setup Weights & Biases logging."""
        try:
            wandb.init(
                project=project,
                name=f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config or {}
            )
            self.wandb_run = wandb.run
            self.info("W&B logging initialized successfully")
        except Exception as e:
            self.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
            self.wandb_run = None
    
    def debug(self, message: str):
        """Log debug message."""
        self.file_logger.debug(message)
        self.console_logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.file_logger.info(message)
        self.console_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.file_logger.warning(message)
        self.console_logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.file_logger.error(message)
        self.console_logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to all configured loggers.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Log to file
        self.file_logger.info(f"Metrics: {json.dumps(metrics)}")
        
        # Log to console (formatted)
        formatted = self.format_metrics(metrics)
        self.console_logger.info(formatted)
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            wandb_metrics = {k: v for k, v in metrics.items() 
                           if k not in ['timestamp', 'step']}
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)
    
    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for console display."""
        # Extract key metrics
        key_metrics = []
        
        if 'accuracy' in metrics:
            key_metrics.append(f"Acc: {metrics['accuracy']:.4f}")
        
        if 'loss' in metrics:
            key_metrics.append(f"Loss: {metrics['loss']:.4f}")
        
        if 'policy_loss' in metrics:
            key_metrics.append(f"P_Loss: {metrics['policy_loss']:.4f}")
        
        if 'value_loss' in metrics:
            key_metrics.append(f"V_Loss: {metrics['value_loss']:.4f}")
        
        if 'step' in metrics:
            key_metrics.insert(0, f"Step: {metrics['step']}")
        
        return " | ".join(key_metrics)
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict, eval_metrics: Optional[Dict] = None):
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            eval_metrics: Optional evaluation metrics
        """
        self.info(f"\n{'='*60}")
        self.info(f"Epoch {epoch} Summary")
        self.info(f"{'='*60}")
        
        # Training metrics
        self.info("Training Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        
        # Evaluation metrics
        if eval_metrics:
            self.info("Evaluation Metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, float):
                    self.info(f"  {key}: {value:.4f}")
                else:
                    self.info(f"  {key}: {value}")
        
        self.info(f"{'='*60}\n")
    
    def log_model_summary(self, model_info: Dict):
        """Log model architecture summary."""
        self.info("\n" + "="*60)
        self.info("Model Summary")
        self.info("="*60)
        
        for key, value in model_info.items():
            self.info(f"{key}: {value}")
        
        self.info("="*60 + "\n")
    
    def log_hyperparameters(self, config: Dict):
        """Log hyperparameters."""
        self.info("\n" + "="*60)
        self.info("Hyperparameters")
        self.info("="*60)
        
        def log_dict(d: Dict, indent: int = 0):
            for key, value in d.items():
                if isinstance(value, dict):
                    self.info(" " * indent + f"{key}:")
                    log_dict(value, indent + 2)
                else:
                    self.info(" " * indent + f"{key}: {value}")
        
        log_dict(config)
        self.info("="*60 + "\n")
    
    def save_metrics(self, filepath: Optional[str] = None):
        """
        Save metrics history to file.
        
        Args:
            filepath: Optional filepath (uses default if not provided)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"metrics_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        self.info(f"Metrics saved to {filepath}")
    
    def log_sample_generation(self, problem: str, solution: str, 
                            is_correct: bool, target_answer: Optional[str] = None):
        """
        Log a sample generation.
        
        Args:
            problem: Problem text
            solution: Generated solution
            is_correct: Whether solution is correct
            target_answer: Optional target answer
        """
        self.info("\n" + "-"*40)
        self.info("Sample Generation")
        self.info("-"*40)
        self.info(f"Problem: {problem[:200]}..." if len(problem) > 200 else f"Problem: {problem}")
        self.info(f"Solution: {solution[:300]}..." if len(solution) > 300 else f"Solution: {solution}")
        if target_answer:
            self.info(f"Target Answer: {target_answer}")
        self.info(f"Correct: {'✓' if is_correct else '✗'}")
        self.info("-"*40 + "\n")
    
    def finish(self):
        """Finalize logging."""
        # Save final metrics
        self.save_metrics()
        
        # Close wandb
        if self.use_wandb and self.wandb_run:
            wandb.finish()
        
        self.info("Logging finished")

class ProgressTracker:
    """Track training progress with ETA estimation."""
    
    def __init__(self, total_steps: int, logger: Logger):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            logger: Logger instance
        """
        self.total_steps = total_steps
        self.logger = logger
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_times = []
    
    def update(self, step: int = 1, metrics: Optional[Dict] = None):
        """
        Update progress.
        
        Args:
            step: Number of steps to advance
            metrics: Optional metrics to log
        """
        self.current_step += step
        current_time = datetime.now()
        
        # Calculate timing
        elapsed = (current_time - self.start_time).total_seconds()
        self.step_times.append(elapsed / self.current_step)
        
        # Keep only recent times for ETA calculation
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        # Calculate ETA
        avg_time_per_step = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_time_per_step
        
        # Format progress message
        progress = self.current_step / self.total_steps
        eta_str = self.format_time(eta_seconds)
        elapsed_str = self.format_time(elapsed)
        
        progress_msg = (
            f"Progress: {self.current_step}/{self.total_steps} "
            f"({progress*100:.1f}%) | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {eta_str}"
        )
        
        if metrics:
            metrics_str = self.logger.format_metrics(metrics)
            progress_msg += f" | {metrics_str}"
        
        self.logger.info(progress_msg)
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into readable time."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

# Convenience functions
def get_logger(name: str = "PAG-ASTAR", **kwargs) -> Logger:
    """Get or create a logger instance."""
    return Logger(name=name, **kwargs)

def log_experiment_start(config: Dict, logger: Logger):
    """Log experiment start."""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT START")
    logger.info("="*60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.log_hyperparameters(config)

def log_experiment_end(logger: Logger, success: bool = True):
    """Log experiment end."""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT END")
    logger.info("="*60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
    logger.info("="*60)
    logger.finish()