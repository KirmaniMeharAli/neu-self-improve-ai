"""
Main entry point for PAG with A*-PO on MATH dataset - GPU Version
"""
import torch
import yaml
import argparse
from transformers import AutoTokenizer
import os

# Set environment variables
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages

from models.policy_model import PolicyModel
from models.value_model import ValueModel
from algorithms.pag import PAG
from data.math_dataset import create_dataloaders
from training.trainer import Trainer

def main(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("PAG with A*-PO on MATH Dataset - GPU Version")
    print("=" * 50)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['train_path']}")
    print(f"Eval Size: {config['dataset']['eval_subset_size']}")
    print("=" * 50)
    
    # Force GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Clear cache
        torch.cuda.empty_cache()
    else:
        print("⚠ WARNING: No GPU detected, falling back to CPU")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, eval_loader = create_dataloaders(config, tokenizer)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Evaluation samples: {len(eval_loader.dataset)}")
    
    # Initialize models on GPU
    print("\nInitializing models on GPU...")
    policy_model = PolicyModel(
        model_name=config['model']['name'],
        device=device
    )
    
    value_model = ValueModel(
        model_name=config['model']['name'],
        hidden_size=1536,  # For Qwen2.5-1.5B
        device=device  # Pass device to constructor
    ).to(device)
    
    # Initialize PAG system
    print("\nInitializing PAG with A*-PO...")
    pag_system = PAG(
        policy_model=policy_model,
        value_model=value_model,
        mcts_config={
            'c_puct': float(config['mcts']['c_puct']),
            'num_simulations': int(config['mcts']['num_simulations']),
            'max_depth': int(config['mcts']['max_depth'])
        },
        astar_config={
            'learning_rate': float(config['astar_po']['learning_rate']),
            'advantage_weight': float(config['astar_po']['advantage_weight']),
            'kl_weight': float(config['astar_po']['kl_weight']),
            'entropy_weight': float(config['astar_po']['entropy_weight']),
            'clip_range': float(config['astar_po']['clip_range']),
            'max_grad_norm': float(config['astar_po']['max_grad_norm'])
        }
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        pag_system=pag_system,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config
    )
    
    # Start training
    print("\nStarting training on GPU...")
    trainer.train()
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PAG with A*-PO on MATH')
    parser.add_argument(
        '--config',
        type=str,
        default='config_gpu.yaml',  # Default to GPU config
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)