"""
Configuration Management Utilities
Author: Student Name
Course: NEU Self-Improving AI
Date: 2025

Handles configuration loading and validation for the MCTS-LLM project.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Supports YAML and JSON formats.
    Falls back to default configuration if no file provided.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
    """
    if config_path is None:
        return get_default_config()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format and load
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    # Validate configuration
    validated_config = validate_config(config)
    
    return validated_config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill in missing configuration values.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration with defaults filled in
    """
    default = get_default_config()
    
    # Recursively merge configurations
    validated = merge_configs(default, config)
    
    # Validate specific constraints
    if validated['mcts']['num_simulations'] < 1:
        raise ValueError("num_simulations must be at least 1")
    
    if validated['mcts']['exploration_constant'] <= 0:
        raise ValueError("exploration_constant must be positive")
    
    if validated['llm']['temperature'] < 0 or validated['llm']['temperature'] > 2:
        logging.warning("LLM temperature outside typical range [0, 2]")
    
    if validated['task']['coherence_weight'] + validated['task']['creativity_weight'] > 1.0:
        logging.warning("Task weights sum to more than 1.0, normalizing...")
        total = validated['task']['coherence_weight'] + validated['task']['creativity_weight']
        validated['task']['coherence_weight'] /= total
        validated['task']['creativity_weight'] /= total
    
    return validated


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.
    
    Args:
        default: Default configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for experiments.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'experiment': {
            'name': 'MCTS-LLM Creative Writing',
            'num_trials': 10,
            'random_seed': 42,
            'save_intermediate': True,
            'output_dir': 'results'
        },
        'mcts': {
            'num_simulations': 100,
            'exploration_constant': 1.414,  # sqrt(2)
            'simulation_depth_limit': 10,
            'time_limit': None,
            'verbose': False
        },
        'llm': {
            'model_name': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 500,
            'top_p': 0.95,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'retry_attempts': 3,
            'retry_delay': 1.0
        },
        'task': {
            'genre': 'fantasy',
            'target_length': 5,
            'coherence_weight': 0.6,
            'creativity_weight': 0.4,
            'max_depth': 7,
            'evaluation_criteria': [
                'narrative_coherence',
                'character_development',
                'plot_structure',
                'thematic_depth',
                'originality'
            ]
        },
        'prompts': [
            "Create a fantasy story about a reluctant hero discovering their destiny",
            "Write a mystery story set in a haunted library with ancient secrets",
            "Develop a sci-fi narrative about time paradoxes and their consequences",
            "Create a romance story with an unexpected twist that changes everything",
            "Write a thriller about corporate espionage in the tech industry"
        ],
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': None  # If specified, log to file
        }
    }


def save_config(config: Dict[str, Any], path: str, format: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
        format: Output format ('yaml' or 'json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_sample_config(output_path: str = "configs/sample_config.yaml") -> None:
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path for sample config file
    """
    config = get_default_config()
    
    # Add comments as a special field (will be ignored by loader)
    config['_comments'] = {
        'description': 'Sample configuration for MCTS-LLM experiments',
        'note': 'Adjust parameters based on your specific requirements',
        'mcts_simulations': 'Higher values give better results but take more time',
        'llm_temperature': 'Lower values make output more deterministic',
        'task_weights': 'coherence_weight + creativity_weight should sum to 1.0'
    }
    
    save_config(config, output_path, format='yaml')
    print(f"Sample configuration saved to: {output_path}")


if __name__ == "__main__":
    # Create sample configuration when run directly
    create_sample_config()