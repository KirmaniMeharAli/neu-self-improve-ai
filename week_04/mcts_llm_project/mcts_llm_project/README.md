mcts_llm_project/
├── src/
│   ├── __init__.py
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── node.py              # MCTS node class
│   │   └── uct_algorithm.py     # UCT implementation
│   ├── llm/
│   │   ├── __init__.py
│   │   └── model_interface.py   # LLM integration
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── creative_writing_task.py  # Story generation
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
├── experiments/
│   ├── __init__.py
│   └── run_experiments.py      # Main experiment runner
├── results/                    # Generated results directory
├── requirements.txt
├── setup.py
└── README.md

# MCTS with LLMs for Creative Writing Generation

**Course**: NEU Self-Improving AI  
**Author**: Mehar 
**Assignment**: MCTS-UCT Implementation with LLM Integration

## Project Overview

This project implements Monte Carlo Tree Search (MCTS) with Upper Confidence Bound for Trees (UCT) algorithm, combined with Large Language Models (LLMs) for creative writing generation. The implementation is based on the research paper "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" (Zhou et al., ICML 2024), applying the technique to a novel domain of creative story generation instead of the paper's original focus on mathematical reasoning and logic puzzles.

## Assignment Requirements Met

✅ **Implemented MCTS-UCT** following Sutton & Barto Chapter 13  
✅ **Selected qualifying paper**: LATS from ICML 2024 (top-tier conference)  
✅ **Applied to different task**: Creative writing (vs. paper's math/reasoning tasks)  
✅ **Realistic application**: Story generation (not a gym environment)  

## Technical Implementation

### Core Components

#### 1. MCTS-UCT Algorithm (`src/mcts/`)
- **node.py**: Tree node implementation with UCB1 selection
- **uct_algorithm.py**: Complete UCT algorithm with four phases:
  - Selection: Navigate tree using UCB1 formula
  - Expansion: Add new nodes to tree
  - Simulation: Evaluate leaf nodes
  - Backpropagation: Update statistics

**UCB1 Formula**: `UCB1 = vi + c * sqrt(ln(N) / ni)`
- vi = average value of child i
- c = exploration constant (default √2)
- N = parent visit count
- ni = child visit count

#### 2. LLM Integration (`src/llm/`)
- **model_interface.py**: LLM interface following LATS approach
  - Value estimation through LLM reasoning
  - Action generation via prompting
  - Reflection mechanism for trajectory analysis
  - Caching for efficiency

#### 3. Creative Writing Task (`src/tasks/`)
- **creative_writing_task.py**: Novel application domain
  - Story state representation
  - Coherence and creativity evaluation
  - Progressive story construction
  - Genre and length constraints

#### 4. Experiment Framework (`experiments/`)
- **run_experiments.py**: Comparative analysis system
  - MCTS-LLM implementation
  - Pure LLM baseline
  - Random MCTS baseline
  - Statistical significance testing
  - Visualization generation

#Usage
#Basic Usage

from src.tasks.creative_writing_task import CreativeWritingTask
from src.llm.model_interface import LLMConfig

# Configure LLM
llm_config = LLMConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Create task
task = CreativeWritingTask(llm_config)

# Run MCTS search
result = task.run_mcts_search(
    initial_prompt="Create a fantasy story about a brave knight",
    num_simulations=100
)

print(result['story']['outline'])


## Algorithm Details

### MCTS-UCT Implementation

The implementation follows the standard MCTS framework:

1. **Initialization**: Create root node with initial state
2. **Iteration Loop** (repeat for n simulations):
   - **Selection**: Traverse tree using UCB1 to select promising nodes
   - **Expansion**: Add new child node if not fully expanded
   - **Simulation**: Evaluate leaf node using LLM or rollout
   - **Backpropagation**: Update statistics along path to root
3. **Action Selection**: Choose action with highest visit count

### LLM Integration (LATS Approach)

Following the LATS paper methodology:

- **Action Generation**: LLM generates candidate story actions
- **Value Estimation**: LLM evaluates state quality through reasoning
- **Reflection**: LLM analyzes trajectories for improvement insights

### Creative Writing Task

Story generation process:

1. Start with initial prompt/genre
2. Progressively add story elements:
   - Characters
   - Setting
   - Conflict
   - Plot points
   - Resolution
3. Evaluate coherence and creativity at each step
4. Use MCTS to find optimal story path


#Running Experiments

# Set working directory to project root
import os
os.chdir('C:\\Users\\mehar\\Downloads\\Week_04\\mcts_llm_project\\mcts_llm_project')

# Run experiment
%run experiments/run_experiments.py --trials 3 --simulations 50


## Experimental Results

The framework compares three approaches:

| Method | Description | Strengths | Weaknesses |
|--------|------------|-----------|------------|
| MCTS-LLM | MCTS with LLM value estimation | Balances exploration/exploitation | Computational overhead |
| Pure LLM | Direct generation | Fast | May lack diversity |
| Random MCTS | MCTS with random evaluation | Good exploration | Poor value estimates |

### Metrics
- **Coherence Score**: Narrative consistency
- **Creativity Score**: Originality and engagement
- **Overall Score**: Weighted combination
- **Execution Time**: Computational efficiency

## Key Innovations

1. **Novel Domain Application**: Applies MCTS+LLM to creative writing instead of mathematical reasoning
2. **Structured Story Generation**: Progressive construction with coherence tracking
3. **Hybrid Evaluation**: Combines LLM reasoning with heuristic metrics
4. **Comprehensive Framework**: Complete experimental pipeline with statistical analysis

## Limitations & Future Work

### Current Limitations
- Mock LLM implementation (for demonstration)
- Simplified evaluation metrics
- Limited to story outlines (not full narratives)

### Potential Improvements
- Integration with real LLM APIs (OpenAI, Anthropic)
- Advanced evaluation metrics (BLEU, perplexity, human evaluation)
- Extended to full story generation
- Multi-agent collaboration
- Domain-specific fine-tuning


## References

1. **Zhou, A., et al. (2024)**. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models." *Proceedings of ICML 2024*.

2. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 13: Policy Gradient Methods.

3. **Kocsis, L., & Szepesvári, C. (2006)**. "Bandit based Monte-Carlo Planning." *ECML 2006*.

4. **Browne, C., et al. (2012)**. "A Survey of Monte Carlo Tree Search Methods." *IEEE Transactions on Computational Intelligence and AI in Games*.











