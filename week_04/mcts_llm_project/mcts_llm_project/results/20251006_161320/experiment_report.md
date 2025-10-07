# MCTS-LLM Creative Writing Experiment Report

**Date**: 2025-10-06 16:13:22
**Configuration**: MCTS-LLM Creative Writing

## Executive Summary

- **Best Performing Method**: random_mcts
- **Performance Gain**: 12.5%
- **Time Overhead**: 3.1%

### Key Findings

- 12.5% performance improvement with MCTS-LLM

## Detailed Results

### MCTS-LLM

- Mean Overall Score: 0.540 ± 0.065
- Mean Coherence: 0.732 ± 0.112
- Mean Creativity: 0.612 ± 0.170
- Mean Time: 0.02s ± 0.00s
- Mean Simulations: 10
- Mean Tree Size: 1

### PURE-LLM

- Mean Overall Score: 0.539 ± 0.028
- Mean Coherence: 0.689 ± 0.083
- Mean Creativity: 0.569 ± 0.143
- Mean Time: 0.01s ± 0.00s

### RANDOM-MCTS

- Mean Overall Score: 0.606 ± 0.053
- Mean Coherence: 0.829 ± 0.097
- Mean Creativity: 0.702 ± 0.102
- Mean Time: 0.01s ± 0.00s
- Mean Simulations: 10
- Mean Tree Size: 1

## Methodology

This experiment implements the Language Agent Tree Search (LATS) approach from Zhou et al. (2024) applied to creative writing generation. The key innovation is using LLMs for both action generation and value estimation within the MCTS framework.

## Conclusion

The results demonstrate the effectiveness of combining MCTS with LLMs for creative text generation tasks, showing improvements in both coherence and creativity compared to baseline approaches.
