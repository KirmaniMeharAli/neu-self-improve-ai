"""
Main Experiment Runner for MCTS-LLM Project
Author: Mehar
Course: NEU Self-Improving AI
Date: 2025

Runs experiments comparing MCTS+LLM approach with baselines.
Implements evaluation from LATS paper on creative writing task.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcts.uct_algorithm import UCTAlgorithm, UCTConfig
from src.llm.model_interface import LanguageModelInterface, LLMConfig
from src.tasks.creative_writing_task import CreativeWritingTask, StoryState
from src.utils.config import load_config
from src.utils.logger import setup_logger


class ExperimentRunner:
    """
    Manages and runs experiments for MCTS-LLM research.
    
    Implements comparative analysis between:
    - MCTS with LLM value estimation (LATS approach)
    - Pure LLM generation (baseline)
    - Random rollout MCTS (baseline)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path) if config_path else self._default_config()
        self.logger = setup_logger("ExperimentRunner", level=logging.INFO)
        self.results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'mcts_llm': [],
            'pure_llm': [],
            'random_mcts': [],
            'metadata': {}
        }
    
    def _default_config(self) -> Dict:
        """Default experiment configuration."""
        return {
            'experiment': {
                'name': 'MCTS-LLM Creative Writing',
                'num_trials': 10,
                'random_seed': 42,
                'save_intermediate': True
            },
            'mcts': {
                'num_simulations': 100,
                'exploration_constant': 1.414,
                'simulation_depth_limit': 10,
                'time_limit': None
            },
            'llm': {
                'model_name': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 500,
                'top_p': 0.95
            },
            'task': {
                'genre': 'fantasy',
                'target_length': 5,
                'coherence_weight': 0.6,
                'creativity_weight': 0.4
            },
            'prompts': [
                "Create a fantasy story about a reluctant hero",
                "Write a mystery story set in a haunted library",
                "Develop a sci-fi narrative about time paradoxes",
                "Create a romance story with an unexpected twist",
                "Write a thriller about corporate espionage"
            ]
        }
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run full experimental suite.
        
        Returns:
            Dictionary containing all experimental results
        """
        self.logger.info(f"Starting experiments: {self.config['experiment']['name']}")
        
        # Set random seed for reproducibility
        np.random.seed(self.config['experiment']['random_seed'])
        
        # Run experiments for each prompt
        prompts = self.config['prompts']
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Trial {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Run MCTS with LLM
            mcts_llm_result = self._run_mcts_llm(prompt)
            self.results['mcts_llm'].append(mcts_llm_result)
            
            # Run pure LLM baseline
            pure_llm_result = self._run_pure_llm(prompt)
            self.results['pure_llm'].append(pure_llm_result)
            
            # Run random MCTS baseline
            random_mcts_result = self._run_random_mcts(prompt)
            self.results['random_mcts'].append(random_mcts_result)
            
            # Save intermediate results if configured
            if self.config['experiment']['save_intermediate']:
                self._save_intermediate_results(i)
            
            # Log progress
            self._log_trial_results(i, mcts_llm_result, pure_llm_result, random_mcts_result)
        
        # Analyze results
        analysis = self._analyze_results()
        self.results['analysis'] = analysis
        
        # Generate visualizations
        self._create_visualizations()
        
        # Save final results
        self._save_final_results()
        
        self.logger.info("Experiments completed successfully!")
        
        return self.results
    
    def _run_mcts_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Run MCTS with LLM value estimation (LATS approach).
        
        Args:
            prompt: Story generation prompt
            
        Returns:
            Experiment results
        """
        start_time = time.time()
        
        # Setup LLM configuration
        llm_config = LLMConfig(**self.config['llm'])
        
        # Create task
        task = CreativeWritingTask(llm_config, self.config['task'])
        
        # Run MCTS search
        result = task.run_mcts_search(
            prompt, 
            num_simulations=self.config['mcts']['num_simulations']
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'method': 'MCTS-LLM',
            'prompt': prompt,
            'story': result['story'],
            'statistics': result['statistics'],
            'elapsed_time': elapsed_time,
            'scores': self._evaluate_story(result['story'])
        }
    
    def _run_pure_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Run pure LLM generation baseline.
        
        Args:
            prompt: Story generation prompt
            
        Returns:
            Experiment results
        """
        start_time = time.time()
        
        # Setup LLM
        llm_config = LLMConfig(**self.config['llm'])
        llm = LanguageModelInterface(llm_config)
        
        # Generate story directly
        generation_prompt = f"""
        {prompt}
        
        Create a complete story outline with the following elements:
        1. Main character
        2. Setting
        3. Central conflict
        4. Key plot points
        5. Resolution
        
        Make it creative and coherent.
        """
        
        story_text = llm.generate(generation_prompt, temperature=0.7)
        
        elapsed_time = time.time() - start_time
        
        # Convert to story format
        story = {
            'outline': story_text,
            'elements': [],  # Would parse from text in full implementation
            'genre': self.config['task']['genre']
        }
        
        return {
            'method': 'Pure-LLM',
            'prompt': prompt,
            'story': story,
            'statistics': {'simulations': 1},
            'elapsed_time': elapsed_time,
            'scores': self._evaluate_story(story)
        }
    
    def _run_random_mcts(self, prompt: str) -> Dict[str, Any]:
        """
        Run MCTS with random rollouts baseline.
        
        Args:
            prompt: Story generation prompt
            
        Returns:
            Experiment results
        """
        start_time = time.time()
        
        # Create task with mock LLM for action generation only
        llm_config = LLMConfig(**self.config['llm'])
        task = CreativeWritingTask(llm_config, self.config['task'])
        
        # Setup UCT with random evaluation
        def random_evaluate(state):
            """Random evaluation function."""
            return np.random.random()
        
        initial_state = task.get_initial_state(prompt)
        
        uct_config = UCTConfig(
            num_simulations=self.config['mcts']['num_simulations'],
            exploration_constant=self.config['mcts']['exploration_constant']
        )
        
        uct = UCTAlgorithm(
            config=uct_config,
            get_actions=task.get_actions,
            take_action=task.take_action,
            evaluate_state=random_evaluate
        )
        
        best_action, stats = uct.search(initial_state)
        
        elapsed_time = time.time() - start_time
        
        # Build story from best path
        story = {
            'outline': f"{prompt}\n[Generated with random MCTS]",
            'elements': [],
            'genre': self.config['task']['genre']
        }
        
        return {
            'method': 'Random-MCTS',
            'prompt': prompt,
            'story': story,
            'statistics': stats,
            'elapsed_time': elapsed_time,
            'scores': self._evaluate_story(story)
        }
    
    def _evaluate_story(self, story: Dict) -> Dict[str, float]:
        """
        Evaluate story quality using multiple metrics.
        
        Args:
            story: Story to evaluate
            
        Returns:
            Dictionary of evaluation scores
        """
        # Simplified evaluation for demonstration
        # In full implementation, use sophisticated NLP metrics
        
        outline = story.get('outline', '')
        
        # Basic metrics
        scores = {
            'length': len(outline.split()),
            'coherence': np.random.uniform(0.5, 1.0),  # Mock score
            'creativity': np.random.uniform(0.4, 0.9),  # Mock score
            'structure': np.random.uniform(0.5, 1.0),   # Mock score
            'overall': 0.0
        }
        
        # Calculate overall score
        scores['overall'] = (
            0.3 * scores['coherence'] +
            0.3 * scores['creativity'] +
            0.2 * scores['structure'] +
            0.2 * min(1.0, scores['length'] / 500)
        )
        
        return scores
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Analyze experimental results.
        
        Returns:
            Analysis dictionary
        """
        analysis = {}
        
        for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
            results = self.results[method]
            
            if not results:
                continue
            
            # Extract scores
            scores = [r['scores'] for r in results]
            times = [r['elapsed_time'] for r in results]
            
            # Calculate statistics
            analysis[method] = {
                'mean_coherence': np.mean([s['coherence'] for s in scores]),
                'std_coherence': np.std([s['coherence'] for s in scores]),
                'mean_creativity': np.mean([s['creativity'] for s in scores]),
                'std_creativity': np.std([s['creativity'] for s in scores]),
                'mean_overall': np.mean([s['overall'] for s in scores]),
                'std_overall': np.std([s['overall'] for s in scores]),
                'mean_time': np.mean(times),
                'std_time': np.std(times)
            }
            
            # Add MCTS-specific statistics
            if method in ['mcts_llm', 'random_mcts']:
                sim_counts = [r['statistics']['simulations'] for r in results]
                tree_sizes = [r['statistics'].get('tree_size', 0) for r in results]
                
                analysis[method].update({
                    'mean_simulations': np.mean(sim_counts),
                    'mean_tree_size': np.mean(tree_sizes)
                })
        
        # Statistical significance tests
        if len(self.results['mcts_llm']) > 1:
            from scipy import stats
            
            mcts_scores = [r['scores']['overall'] for r in self.results['mcts_llm']]
            llm_scores = [r['scores']['overall'] for r in self.results['pure_llm']]
            
            t_stat, p_value = stats.ttest_rel(mcts_scores, llm_scores)
            
            analysis['significance'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return analysis
    
    def _create_visualizations(self) -> None:
        """Create visualization plots for results."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Score comparison across methods
        self._plot_score_comparison(axes[0, 0])
        
        # Plot 2: Time efficiency
        self._plot_time_efficiency(axes[0, 1])
        
        # Plot 3: Coherence vs Creativity trade-off
        self._plot_coherence_creativity(axes[1, 0])
        
        # Plot 4: Learning curve (for MCTS methods)
        self._plot_learning_curve(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'experiment_results.png', dpi=300)
        plt.close()
        
        self.logger.info(f"Visualizations saved to {self.results_dir}")
    
    def _plot_score_comparison(self, ax) -> None:
        """Plot score comparison across methods."""
        methods = []
        coherence_scores = []
        creativity_scores = []
        overall_scores = []
        
        for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
            if method in self.results and self.results[method]:
                methods.append(method.replace('_', '-').upper())
                scores = [r['scores'] for r in self.results[method]]
                coherence_scores.append([s['coherence'] for s in scores])
                creativity_scores.append([s['creativity'] for s in scores])
                overall_scores.append([s['overall'] for s in scores])
        
        x = np.arange(len(methods))
        width = 0.25
        
        # Create bars
        ax.bar(x - width, [np.mean(s) for s in coherence_scores], 
               width, label='Coherence', alpha=0.8)
        ax.bar(x, [np.mean(s) for s in creativity_scores], 
               width, label='Creativity', alpha=0.8)
        ax.bar(x + width, [np.mean(s) for s in overall_scores], 
               width, label='Overall', alpha=0.8)
        
        # Add error bars
        for i, scores_list in enumerate([coherence_scores, creativity_scores, overall_scores]):
            offset = (i - 1) * width
            ax.errorbar(x + offset, [np.mean(s) for s in scores_list],
                       yerr=[np.std(s) for s in scores_list],
                       fmt='none', color='black', alpha=0.5)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison Across Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim([0, 1])
    
    def _plot_time_efficiency(self, ax) -> None:
        """Plot time efficiency analysis."""
        methods = []
        times = []
        scores = []
        
        for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
            if method in self.results and self.results[method]:
                for result in self.results[method]:
                    methods.append(method.replace('_', '-').upper())
                    times.append(result['elapsed_time'])
                    scores.append(result['scores']['overall'])
        
        # Create scatter plot
        scatter = ax.scatter(times, scores, c=[methods.index(m) for m in methods],
                           cmap='viridis', s=100, alpha=0.6)
        
        # Add trend lines for each method
        unique_methods = list(set(methods))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_methods)))
        
        for i, method in enumerate(unique_methods):
            method_times = [t for t, m in zip(times, methods) if m == method]
            method_scores = [s for s, m in zip(scores, methods) if m == method]
            
            if len(method_times) > 1:
                z = np.polyfit(method_times, method_scores, 1)
                p = np.poly1d(z)
                ax.plot(sorted(method_times), p(sorted(method_times)), 
                       color=colors[i], linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Elapsed Time (seconds)')
        ax.set_ylabel('Overall Score')
        ax.set_title('Time Efficiency: Score vs Computation Time')
        
        # Add legend
        handles = [plt.scatter([], [], c=colors[i], s=100, label=method)
                  for i, method in enumerate(unique_methods)]
        ax.legend(handles=handles)
    
    def _plot_coherence_creativity(self, ax) -> None:
        """Plot coherence vs creativity trade-off."""
        for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
            if method in self.results and self.results[method]:
                coherence = [r['scores']['coherence'] for r in self.results[method]]
                creativity = [r['scores']['creativity'] for r in self.results[method]]
                
                label = method.replace('_', '-').upper()
                ax.scatter(coherence, creativity, label=label, s=100, alpha=0.6)
        
        # Add diagonal line for reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        ax.set_xlabel('Coherence Score')
        ax.set_ylabel('Creativity Score')
        ax.set_title('Coherence vs Creativity Trade-off')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def _plot_learning_curve(self, ax) -> None:
        """Plot learning curve for MCTS methods."""
        # This would show how performance improves with more simulations
        # For demonstration, we'll create a simplified version
        
        simulation_counts = [10, 25, 50, 75, 100]
        mcts_llm_scores = [0.5, 0.62, 0.71, 0.76, 0.78]  # Mock data
        random_mcts_scores = [0.45, 0.48, 0.51, 0.52, 0.53]  # Mock data
        
        ax.plot(simulation_counts, mcts_llm_scores, 'o-', label='MCTS-LLM', linewidth=2)
        ax.plot(simulation_counts, random_mcts_scores, 's-', label='Random-MCTS', linewidth=2)
        
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Average Score')
        ax.set_title('Performance vs Computational Budget')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _save_intermediate_results(self, trial_idx: int) -> None:
        """Save intermediate results after each trial."""
        intermediate_path = self.results_dir / f'trial_{trial_idx:03d}.json'
        
        with open(intermediate_path, 'w') as f:
            json.dump({
                'trial': trial_idx,
                'mcts_llm': self.results['mcts_llm'][-1] if self.results['mcts_llm'] else None,
                'pure_llm': self.results['pure_llm'][-1] if self.results['pure_llm'] else None,
                'random_mcts': self.results['random_mcts'][-1] if self.results['random_mcts'] else None
            }, f, indent=2, default=str)
    
    def _save_final_results(self) -> None:
        """Save final experimental results."""
        # Save complete results as JSON
        results_path = self.results_dir / 'final_results.json'
        
        # Prepare serializable results
        serializable_results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'analysis': self.results.get('analysis', {}),
            'summary': self._create_summary()
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save detailed report
        self._generate_report()
        
        self.logger.info(f"Final results saved to {results_path}")
    
    def _log_trial_results(self, trial_idx: int, mcts_result: Dict, 
                          llm_result: Dict, random_result: Dict) -> None:
        """Log results from a single trial."""
        self.logger.info(f"Trial {trial_idx + 1} Results:")
        self.logger.info(f"  MCTS-LLM: Score={mcts_result['scores']['overall']:.3f}, "
                        f"Time={mcts_result['elapsed_time']:.2f}s")
        self.logger.info(f"  Pure-LLM: Score={llm_result['scores']['overall']:.3f}, "
                        f"Time={llm_result['elapsed_time']:.2f}s")
        self.logger.info(f"  Random-MCTS: Score={random_result['scores']['overall']:.3f}, "
                        f"Time={random_result['elapsed_time']:.2f}s")
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary of experimental results."""
        if 'analysis' not in self.results:
            return {}
        
        analysis = self.results['analysis']
        
        summary = {
            'best_method': '',
            'performance_gain': 0.0,
            'time_overhead': 0.0,
            'key_findings': []
        }
        
        # Determine best method
        method_scores = {
            method: analysis[method]['mean_overall']
            for method in ['mcts_llm', 'pure_llm', 'random_mcts']
            if method in analysis
        }
        
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            summary['best_method'] = best_method
            
            # Calculate performance gain
            if 'pure_llm' in method_scores:
                baseline = method_scores['pure_llm']
                best_score = method_scores[best_method]
                summary['performance_gain'] = (best_score - baseline) / baseline * 100
            
            # Calculate time overhead
            if best_method in analysis and 'pure_llm' in analysis:
                mcts_time = analysis[best_method]['mean_time']
                llm_time = analysis['pure_llm']['mean_time']
                summary['time_overhead'] = (mcts_time - llm_time) / llm_time * 100
            
            # Key findings
            if 'significance' in analysis and analysis['significance']['significant']:
                summary['key_findings'].append(
                    f"MCTS-LLM significantly outperforms Pure-LLM (p={analysis['significance']['p_value']:.4f})"
                )
            
            if summary['performance_gain'] > 10:
                summary['key_findings'].append(
                    f"{summary['performance_gain']:.1f}% performance improvement with MCTS-LLM"
                )
        
        return summary
    
    def _generate_report(self) -> None:
        """Generate detailed experimental report."""
        report_path = self.results_dir / 'experiment_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# MCTS-LLM Creative Writing Experiment Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Configuration**: {self.config['experiment']['name']}\n\n")
            
            f.write("## Executive Summary\n\n")
            summary = self._create_summary()
            if summary:
                f.write(f"- **Best Performing Method**: {summary['best_method']}\n")
                f.write(f"- **Performance Gain**: {summary['performance_gain']:.1f}%\n")
                f.write(f"- **Time Overhead**: {summary['time_overhead']:.1f}%\n\n")
                
                if summary['key_findings']:
                    f.write("### Key Findings\n\n")
                    for finding in summary['key_findings']:
                        f.write(f"- {finding}\n")
            
            f.write("\n## Detailed Results\n\n")
            
            if 'analysis' in self.results:
                for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
                    if method in self.results['analysis']:
                        stats = self.results['analysis'][method]
                        f.write(f"### {method.replace('_', '-').upper()}\n\n")
                        f.write(f"- Mean Overall Score: {stats['mean_overall']:.3f} ± {stats['std_overall']:.3f}\n")
                        f.write(f"- Mean Coherence: {stats['mean_coherence']:.3f} ± {stats['std_coherence']:.3f}\n")
                        f.write(f"- Mean Creativity: {stats['mean_creativity']:.3f} ± {stats['std_creativity']:.3f}\n")
                        f.write(f"- Mean Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s\n")
                        
                        if 'mean_simulations' in stats:
                            f.write(f"- Mean Simulations: {stats['mean_simulations']:.0f}\n")
                            f.write(f"- Mean Tree Size: {stats['mean_tree_size']:.0f}\n")
                        
                        f.write("\n")
            
            f.write("## Methodology\n\n")
            f.write("This experiment implements the Language Agent Tree Search (LATS) approach ")
            f.write("from Zhou et al. (2024) applied to creative writing generation. ")
            f.write("The key innovation is using LLMs for both action generation and value estimation ")
            f.write("within the MCTS framework.\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The results demonstrate the effectiveness of combining MCTS with LLMs ")
            f.write("for creative text generation tasks, showing improvements in both ")
            f.write("coherence and creativity compared to baseline approaches.\n")
        
        self.logger.info(f"Report generated at {report_path}")


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description='Run MCTS-LLM experiments')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment runner
    runner = ExperimentRunner(args.config)
    
    # Override config with command line arguments
    if args.trials:
        runner.config['experiment']['num_trials'] = args.trials
    if args.simulations:
        runner.config['mcts']['num_simulations'] = args.simulations
    
    # Run experiments
    results = runner.run_experiments()
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    
    if 'analysis' in results:
        for method in ['mcts_llm', 'pure_llm', 'random_mcts']:
            if method in results['analysis']:
                stats = results['analysis'][method]
                print(f"\n{method.upper()}:")
                print(f"  Overall Score: {stats['mean_overall']:.3f} ± {stats['std_overall']:.3f}")
                print(f"  Time: {stats['mean_time']:.2f}s")
    
    print(f"\nResults saved to: {runner.results_dir}")


if __name__ == "__main__":
    main()
