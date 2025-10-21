"""
Setup script to install all required packages for PAG with A*-PO
Run this file first in Spyder or Anaconda Prompt
"""

import subprocess
import sys
import os

print("="*70)
print("PAG WITH A*-PO - ENVIRONMENT SETUP")
print("="*70)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print("="*70)

def install_package(package_name):
    """Install a package using pip"""
    try:
        print(f"\nInstalling {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

# Required packages
packages = [
    ("torch", "torch"),  # PyTorch
    ("transformers", "transformers"),  # Hugging Face Transformers
    ("datasets", "datasets"),  # Hugging Face Datasets
    ("numpy", "numpy"),  # NumPy (usually pre-installed)
    ("pandas", "pandas"),  # Pandas (usually pre-installed)
    ("tqdm", "tqdm"),  # Progress bars
    ("pyyaml", "yaml"),  # YAML support
    ("sympy", "sympy"),  # Symbolic math
    ("accelerate", "accelerate"),  # Training acceleration
    ("sentencepiece", "sentencepiece"),  # Tokenization
]

# Optional packages (not critical)
optional_packages = [
    ("wandb", "wandb"),  # Weights & Biases logging
]

print("\nChecking required packages...")
print("-"*40)

missing_packages = []
for package_install_name, package_import_name in packages:
    if check_package(package_import_name):
        print(f"✓ {package_install_name} is already installed")
    else:
        print(f"✗ {package_install_name} is NOT installed")
        missing_packages.append(package_install_name)

print("\nChecking optional packages...")
print("-"*40)

missing_optional = []
for package_install_name, package_import_name in optional_packages:
    if check_package(package_import_name):
        print(f"✓ {package_install_name} is already installed")
    else:
        print(f"○ {package_install_name} is NOT installed (optional)")
        missing_optional.append(package_install_name)

# Install missing packages
if missing_packages:
    print("\n" + "="*70)
    print("INSTALLING MISSING REQUIRED PACKAGES")
    print("="*70)
    print(f"Need to install: {', '.join(missing_packages)}")
    
    response = input("\nDo you want to install these packages? (y/n): ").lower()
    if response == 'y':
        for package in missing_packages:
            install_package(package)
    else:
        print("\n⚠ Warning: Cannot proceed without required packages!")
        sys.exit(1)

# Ask about optional packages
if missing_optional:
    print("\n" + "="*70)
    print("OPTIONAL PACKAGES")
    print("="*70)
    print(f"Optional packages not installed: {', '.join(missing_optional)}")
    response = input("\nDo you want to install optional packages? (y/n): ").lower()
    if response == 'y':
        for package in missing_optional:
            install_package(package)

# Final check
print("\n" + "="*70)
print("FINAL VERIFICATION")
print("="*70)

all_good = True
for package_install_name, package_import_name in packages:
    if check_package(package_import_name):
        print(f"✓ {package_install_name}")
    else:
        print(f"✗ {package_install_name} - STILL MISSING!")
        all_good = False

if all_good:
    print("\n✓✓✓ ALL REQUIRED PACKAGES INSTALLED! ✓✓✓")
    print("\nYou can now run the main.py file!")
    print("\nNext steps:")
    print("1. Make sure 'competition_math.csv' is in the current directory")
    print("2. Run test_manual.py to verify everything works")
    print("3. Run main.py to start training")
else:
    print("\n✗✗✗ SOME PACKAGES ARE STILL MISSING ✗✗✗")
    print("\nPlease install missing packages manually using:")
    print("conda install <package_name> or pip install <package_name>")

print("\n" + "="*70)