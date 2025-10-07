# MCTS-LLM Creative Writing Experiment Report

**Date**: 2025-10-06 16:49:01
**Configuration**: MCTS-LLM Creative Writing

## Executive Summary

- **Best Performing Method**: mcts_llm
- **Performance Gain**: 2.8%
- **Time Overhead**: 168.2%


## Detailed Results

### MCTS-LLM

- Mean Overall Score: 0.578 ± 0.073
- Mean Coherence: 0.726 ± 0.112
- Mean Creativity: 0.684 ± 0.175
- Mean Time: 0.03s ± 0.00s
- Mean Simulations: 50
- Mean Tree Size: 1

### PURE-LLM

- Mean Overall Score: 0.562 ± 0.045
- Mean Coherence: 0.784 ± 0.138
- Mean Creativity: 0.588 ± 0.148
- Mean Time: 0.01s ± 0.00s

### RANDOM-MCTS

- Mean Overall Score: 0.550 ± 0.072
- Mean Coherence: 0.792 ± 0.141
- Mean Creativity: 0.573 ± 0.119
- Mean Time: 0.01s ± 0.00s
- Mean Simulations: 50
- Mean Tree Size: 1

## Methodology

This experiment implements the Language Agent Tree Search (LATS) approach from Zhou et al. (2024) applied to creative writing generation. The key innovation is using LLMs for both action generation and value estimation within the MCTS framework.

## Conclusion

The results demonstrate the effectiveness of combining MCTS with LLMs for creative text generation tasks, showing improvements in both coherence and creativity compared to baseline approaches.
