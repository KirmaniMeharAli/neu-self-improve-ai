# Week 01 - Multi-Armed Bandit Assignment

## Project Structure

```
week_01/
├── environment.py       # Bandit environment implementation
├── agents.py           # Agent implementations (ε-greedy, gradient)
├── experiment.py       # Experiment runner and data collection
├── visualization.py    # Plotting and visualization utilities
├── main.py            # Main execution script
├── figure_2_2_reproduction.png  # Generated figure
└── README.md          # This file
```

## Overview

This project implements a complete reinforcement learning framework for the multi-armed bandit problem, reproducing Figure 2.2 from Sutton & Barto's textbook with additional gradient policy learning methods.

## Implementation Details

### Core Components

1. **Environment Module** (`environment.py`)
   - `BanditEnvironment`: K-armed bandit with stationary reward distributions
   - True action values sampled from N(0,1)
   - Reward generation with Gaussian noise

2. **Agents Module** (`agents.py`)
   - `BaseAgent`: Abstract base class for all agents
   - `EpsilonGreedyAgent`: ε-greedy action selection with sample averaging
   - `GradientBanditAgent`: Softmax policy with gradient ascent on preferences
   - Support for baseline variance reduction

3. **Experiment Module** (`experiment.py`)
   - `ExperimentRunner`: Manages multiple simulations
   - Parallel execution capability
   - Statistical analysis and result aggregation

4. **Visualization Module** (`visualization.py`)
   - `Visualizer`: Creates publication-quality plots
   - Reproduces Figure 2.2 format
   - Additional analysis visualizations

### Algorithms Implemented

#### Epsilon-Greedy Methods
- **ε = 0** (pure greedy)
- **ε = 0.01**
- **ε = 0.1**

#### Gradient Bandit Methods
- **α = 0.1** without baseline
- **α = 0.4** without baseline
- **α = 0.1** with baseline
- **α = 0.4** with baseline

## How to Run

```bash
# Run the main experiment
python main.py
```

This will:
1. Run 2000 independent simulations for each algorithm
2. Generate Figure 2.2 reproduction
3. Display performance statistics
4. Save results to `figure_2_2_reproduction.png`

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Results Summary

### Performance Comparison (Last 100 Steps)

| Method | Average Reward | % Optimal Actions |
|--------|---------------|------------------|
| Gradient α=0.1 (baseline) | 1.493 | 85.0% |
| Gradient α=0.1 (no baseline) | 1.489 | 83.6% |
| ε=0.1 | 1.361 | 80.4% |
| Gradient α=0.4 (baseline) | 1.439 | 72.3% |
| Gradient α=0.4 (no baseline) | 1.427 | 70.4% |
| ε=0.01 | 1.297 | 58.7% |
| ε=0 (greedy) | 1.017 | 35.4% |

### Key Findings

1. **Gradient methods with baseline achieve best performance** - The baseline reduces variance in gradient estimates, leading to more stable learning
2. **Learning rate trade-off** - Higher learning rates (α=0.4) converge faster but with increased variance
3. **Exploration is crucial** - Pure greedy (ε=0) performs poorly without exploration
4. **Baseline improvement** - Baseline consistently improves gradient method performance

## Mathematical Background

### Epsilon-Greedy Update Rule
```
Q_{n+1}(a) = Q_n(a) + (1/n)[R_n - Q_n(a)]
```

### Gradient Bandit Update Rules
For selected action a = A_t:
```
H_{t+1}(A_t) = H_t(A_t) + α(R_t - baseline)(1 - π_t(A_t))
```

For all other actions a ≠ A_t:
```
H_{t+1}(a) = H_t(a) - α(R_t - baseline)π_t(a)
```

Where π_t(a) is the softmax probability:
```
π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))
```

## Student
Mehar  
NEU Self-Improving AI Course  
Week 01 Assignment