"""
Logging Utilities
Author: Student Name
Course: NEU Self-Improving AI  
Date: 2025

Provides consistent logging setup across the MCTS-LLM project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str) -> logging.Logger:
    """
    Get a logger configured for experiments.
    
    Creates a log file in the results directory with timestamp.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=f"experiment.{experiment_name}",
        level=logging.INFO,
        log_file=log_file,
        format_string='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )


class MCTSLogger:
    """
    Specialized logger for MCTS tree search operations.
    
    Provides structured logging for tree operations with performance metrics.
    """
    
    def __init__(self, name: str = "MCTS", verbose: bool = False):
        """
        Initialize MCTS logger.
        
        Args:
            name: Logger name
            verbose: Enable verbose logging
        """
        level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logger(name, level=level)
        self.metrics = {
            'nodes_expanded': 0,
            'simulations_run': 0,
            'best_value': 0.0
        }
    
    def log_selection(self, node_id: int, ucb_value: float) -> None:
        """Log node selection during tree traversal."""
        self.logger.debug(f"Selected node {node_id} with UCB value {ucb_value:.3f}")
    
    def log_expansion(self, parent_id: int, action: str, child_id: int) -> None:
        """Log node expansion."""
        self.logger.debug(f"Expanded node {parent_id} with action '{action}' -> child {child_id}")
        self.metrics['nodes_expanded'] += 1
    
    def log_simulation(self, node_id: int, result: float) -> None:
        """Log simulation result."""
        self.logger.debug(f"Simulation from node {node_id} returned value {result:.3f}")
        self.metrics['simulations_run'] += 1
        self.metrics['best_value'] = max(self.metrics['best_value'], result)
    
    def log_backpropagation(self, path_length: int, value: float) -> None:
        """Log backpropagation phase."""
        self.logger.debug(f"Backpropagating value {value:.3f} through {path_length} nodes")
    
    def log_iteration_summary(self, iteration: int, tree_size: int) -> None:
        """Log summary after each MCTS iteration."""
        self.logger.info(
            f"Iteration {iteration}: Tree size={tree_size}, "
            f"Simulations={self.metrics['simulations_run']}, "
            f"Best value={self.metrics['best_value']:.3f}"
        )
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()


class ExperimentTracker:
    """
    Track and log experiment progress with rich formatting.
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.logger = get_experiment_logger(experiment_name)
        self.start_time = None
        self.checkpoints = []
    
    def start(self) -> None:
        """Mark experiment start."""
        self.start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT STARTED: {self.experiment_name}")
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info("="*60)
    
    def checkpoint(self, name: str, metrics: Optional[dict] = None) -> None:
        """
        Log a checkpoint with optional metrics.
        
        Args:
            name: Checkpoint name
            metrics: Optional metrics dictionary
        """
        checkpoint_time = datetime.now()
        elapsed = (checkpoint_time - self.start_time).total_seconds() if self.start_time else 0
        
        self.checkpoints.append({
            'name': name,
            'time': checkpoint_time,
            'elapsed': elapsed,
            'metrics': metrics
        })
        
        self.logger.info(f"\n--- Checkpoint: {name} ---")
        self.logger.info(f"Elapsed time: {elapsed:.2f}s")
        
        if metrics:
            self.logger.info("Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def end(self, summary: Optional[dict] = None) -> None:
        """
        Mark experiment end with optional summary.
        
        Args:
            summary: Optional summary dictionary
        """
        end_time = datetime.now()
        total_elapsed = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Total elapsed: {total_elapsed:.2f}s")
        
        if summary:
            self.logger.info("\nSummary:")
            for key, value in summary.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*60)
    
    def log_error(self, error: Exception) -> None:
        """
        Log an error with traceback.
        
        Args:
            error: Exception that occurred
        """
        import traceback
        
        self.logger.error(f"Error occurred: {str(error)}")
        self.logger.error("Traceback:")
        self.logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("test", level=logging.DEBUG)
    logger.info("Test message")
    
    # MCTS logger example
    mcts_logger = MCTSLogger(verbose=True)
    mcts_logger.log_selection(1, 0.95)
    mcts_logger.log_expansion(1, "action_1", 2)
    mcts_logger.log_simulation(2, 0.75)
    mcts_logger.log_iteration_summary(1, 5)
    
    # Experiment tracker example
    tracker = ExperimentTracker("test_experiment")
    tracker.start()
    tracker.checkpoint("initialization", {"loss": 0.5})
    tracker.checkpoint("training", {"loss": 0.3, "accuracy": 0.85})
    tracker.end({"final_accuracy": 0.92})