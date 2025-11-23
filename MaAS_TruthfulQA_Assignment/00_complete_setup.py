"""
00_complete_setup.py
Complete setup script for MaAS TruthfulQA Assignment
Author: Syed Mehar Ali
INFO 6105 - Data Science & Self-Improving AI

This script sets up the environment without requiring full MaAS dependencies
"""

import os
import sys
import subprocess

def main():
    print("="*70)
    print("MaAS × TruthfulQA Assignment Setup")
    print("Author: Syed Mehar Ali")
    print("="*70)
    
    # Step 1: Create directory structure
    print("\n[Step 1] Creating project directories...")
    directories = ['data', 'results', 'visualizations', 'logs']
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created: {dir_name}/")
    
    # Step 2: Check and install required packages
    print("\n[Step 2] Checking required packages...")
    
    required_packages = {
        'datasets': 'For TruthfulQA dataset',
        'pandas': 'For data manipulation',
        'numpy': 'For numerical operations',
        'matplotlib': 'For visualizations',
        'tqdm': 'For progress bars'
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description} [MISSING]")
            missing.append(package)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        for package in missing:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("✓ All packages installed")
    
    # Step 3: Test TruthfulQA access
    print("\n[Step 3] Testing TruthfulQA dataset access...")
    try:
        from datasets import load_dataset
        print("Loading TruthfulQA...")
        dataset = load_dataset("truthful_qa", "generation")
        val_size = len(dataset['validation'])
        print(f"✓ Successfully loaded {val_size} questions")
        
        # Show sample
        sample = dataset['validation'][0]
        print(f"\nSample question from dataset:")
        print(f"  Q: {sample['question'][:80]}...")
        print(f"  A: {sample['best_answer'][:80]}...")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    # Step 4: Create a configuration file
    print("\n[Step 4] Creating configuration file...")
    config = {
        'project_name': 'MaAS_TruthfulQA_Assignment',
        'author': 'Syed Mehar Ali',
        'course': 'INFO 6105',
        'dataset': 'TruthfulQA',
        'num_questions': 100,
        'adaptation_note': 'Using simulated MaAS wrapper due to dependency constraints'
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Created config.json")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nProject Structure:")
    print("  📁 data/           - Dataset files")
    print("  📁 results/        - Experiment results")
    print("  📁 visualizations/ - Architecture diagrams")
    print("  📁 logs/           - Execution logs")
    print("\nIMPORTANT NOTE:")
    print("We're using a simulated MaAS wrapper for this assignment.")
    print("This is documented in config.json as per assignment guidelines.")
    print("\nNext steps:")
    print("1. Run 01_truthfulqa_explorer.py to explore the dataset")
    print("2. Run 02_maas_simulator.py to understand the simulation")
    print("3. Run 03_run_experiment.py to conduct the experiment")
    print("4. Run 04_analyze_results.py to analyze failures/successes")
    print("5. Run 05_generate_report.py to create final submission")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠ Setup incomplete. Please fix errors and run again.")
    else:
        print("\n✅ You're ready to start! Run the files in order.")