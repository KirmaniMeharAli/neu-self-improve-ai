# Week 02 - Self-Improving AI Assignment

**Author:** Mehar  
**Course:** NEU Self-Improving AI  
**References:** Sutton & Barto, Chapters 8-10

## Overview

This assignment implements various reinforcement learning control algorithms on gridworld environments, specifically replicating experiments from Sutton & Barto's textbook.

## Project Structure

```
week_02/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── gridworld.py                      # Basic 4x4 gridworld implementation
├── windy_gridworld.py                # Stochastic windy gridworld
├── algorithms/
│   ├── __init__.py
│   ├── dynamic_programming.py        # DP algorithms (Policy/Value Iteration)
│   ├── monte_carlo.py               # MC control algorithms
│   └── temporal_difference.py        # TD algorithms (SARSA, TD with IS)
├── experiments/
│   ├── __init__.py
│   ├── figure_4_1_replication.py    # Replicates Figure 4.1
│   └── run_all_algorithms.py        # Runs all algorithms for comparison
└── utils/
    ├── __init__.py
    ├── plotting.py                   # Visualization utilities
    └── helpers.py                    # Helper functions
```

## Installation

1. Navigate to week_02 directory:
```bash
cd neu-self-improve-ai/week_02
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

### 1. Figure 4.1 Replication

Replicates the iterative policy evaluation experiment from Sutton & Barto Figure 4.1.

**Run:**
```bash
cd experiments
python figure_4_1_replication.py
```

**What it does:**
- Implements a 4x4 gridworld with terminal states at corners
- Runs iterative policy evaluation for a uniform random policy
- Shows value function at iterations k=0, 1, 2, 3, 10, and ∞
- Generates visualization similar to the textbook figure

### 2. Windy Gridworld Implementation

Implements the stochastic windy gridworld environment with wind effects that push the agent upward.

**Features:**
- 7x10 grid with stochastic wind
- Wind strength varies by column [0,0,0,1,1,1,2,2,1,0]
- Stochastic wind: 80% base effect, 10% +1, 10% no wind
- Start position: (3, 0), Goal position: (3, 7)

### 3. Control Algorithms Comparison

**Run all algorithms:**
```bash
cd experiments
python run_all_algorithms.py
```

**Implemented Algorithms:**

#### Dynamic Programming (DP)
- **Policy Iteration:** Alternates between policy evaluation and improvement
- **Value Iteration:** Combined evaluation and improvement (used for DP control requirement)

#### Monte Carlo (MC)
- **On-Policy MC Control:** Uses epsilon-soft policies with first-visit returns
- **Off-Policy MC Control:** Uses importance sampling for learning from behavior policy

#### Temporal Difference (TD)
- **TD(0) On-Policy Control:** SARSA algorithm
- **TD(0) Off-Policy with Unweighted IS:** TD learning with unweighted importance sampling
- **TD(0) Off-Policy with Weighted IS:** TD learning with weighted importance sampling

## Results Summary

### Algorithm Performance Comparison

| Algorithm | Type | Convergence Speed | Sample Efficiency | Computation Time |
|-----------|------|------------------|-------------------|------------------|
| DP Control | DP | Fast | N/A (model-based) | High |
| MC On-Policy | MC | Moderate | Low | Moderate |
| MC Off-Policy | MC | Slow | Low | Moderate |
| TD(0) On-Policy | TD | Fast | High | Low |
| TD(0) Off-Policy Unweighted IS | TD | Moderate | Medium | Low |
| TD(0) Off-Policy Weighted IS | TD | Moderate | Medium | Low |

### Key Observations

1. **DP algorithms** require complete model knowledge but converge quickly
2. **MC algorithms** are model-free but need complete episodes
3. **TD algorithms** learn online and balance between DP and MC approaches
4. **Importance sampling** allows off-policy learning with corrections for distribution mismatch

## Code Examples

### Running a specific algorithm:

```python
from windy_gridworld import WindyGridworld
from algorithms.temporal_difference import TemporalDifference

# Create environment
env = WindyGridworld(stochastic=True)

# Initialize TD learner
td = TemporalDifference(env, gamma=0.9)

# Run SARSA (TD(0) on-policy)
Q, episode_lengths = td.sarsa(n_episodes=500, alpha=0.5, epsilon=0.1)

# Extract and display policy
policy = td.extract_policy_from_q(Q)
env.render(policy=policy)
```

### Visualizing results:

```python
from utils.plotting import plot_value_function, plot_episode_lengths

# Plot value function
V = np.max(Q, axis=1)  # Get V from Q
plot_value_function(V, env, title="SARSA Value Function")

# Plot learning curve
plot_episode_lengths(episode_lengths, title="SARSA Learning Progress")
```

## Assignment Requirements Checklist

✅ **Replicate the experiment in Figure 4.1**
- Implemented in `experiments/figure_4_1_replication.py`

✅ **Implement Windy Gridworld**
- Implemented in `windy_gridworld.py`

✅ **Implement control algorithms over Windy Gridworld:**
- ✅ DP control → `algorithms/dynamic_programming.py`
- ✅ MC on-policy control → `algorithms/monte_carlo.py`
- ✅ MC off-policy control → `algorithms/monte_carlo.py`
- ✅ TD(0) on-policy control → `algorithms/temporal_difference.py`
- ✅ TD(0) off-policy control with unweighted IS → `algorithms/temporal_difference.py`
- ✅ TD(0) off-policy control with weighted IS → `algorithms/temporal_difference.py`

## File Descriptions

### Core Environment Files

- **`gridworld.py`**: Basic 4x4 gridworld for Figure 4.1 replication
- **`windy_gridworld.py`**: Main environment for control algorithms

### Algorithm Implementations

- **`dynamic_programming.py`**: Model-based planning algorithms (Policy & Value Iteration)
- **`monte_carlo.py`**: Episode-based learning algorithms
- **`temporal_difference.py`**: Online learning algorithms with importance sampling variants

### Experiment Scripts

- **`figure_4_1_replication.py`**: Reproduces textbook figure
- **`run_all_algorithms.py`**: Comprehensive comparison of all algorithms

### Utilities

- **`plotting.py`**: Visualization functions for results
- **`helpers.py`**: Common utility functions

## Dependencies

- numpy >= 1.19.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## Notes

- All algorithms use discount factor γ = 0.9 by default
- Episode limits prevent infinite loops
- Stochastic wind adds realism to the environment
- Code follows professional Python practices with clear documentation

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.)
- Chapters 8-10: Planning and Learning with Tabular Methods
- Note: The fundamental algorithms (DP, MC, TD) are introduced in Chapters 4-6, while Chapters 8-10 cover their integration and advanced topics

## Contact

For questions about this implementation, please contact through the course portal or GitHub issues.