"""
Created on Sep 23 02:04:22 2025

@author: mehar

This module contains functions for creating plots and visualizations.

"""

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Handles visualization of experiment results.
    """
    
    def __init__(self, style='seaborn'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use
        """
        if style:
            plt.style.use(style)
        
        # Define color scheme for different methods
        self.colors = {
            'ε=0 (greedy)': '#2E7D32',  # Green
            'ε=0.01': '#D32F2F',         # Red  
            'ε=0.1': '#1976D2',          # Blue
            'gradient α=0.1 (no baseline)': '#7B1FA2',  # Purple
            'gradient α=0.4 (no baseline)': '#F57C00',  # Orange
            'gradient α=0.1 (baseline)': '#5D4037',     # Brown
            'gradient α=0.4 (baseline)': '#C2185B'      # Pink
        }
        
        # Line styles for different methods
        self.line_styles = {
            'ε=0 (greedy)': '-',
            'ε=0.01': '-',
            'ε=0.1': '-',
            'gradient α=0.1 (no baseline)': '--',
            'gradient α=0.4 (no baseline)': '--',
            'gradient α=0.1 (baseline)': '-',
            'gradient α=0.4 (baseline)': '-'
        }
    
    def create_figure_2_2(self, results_dict, save_path='figure_2_2_reproduction.png'):
        """
        Create reproduction of Figure 2.2 from Sutton & Barto.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary containing experiment results
        save_path : str
            Path to save the figure
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top plot: Average Reward
        ax1 = axes[0]
        self._plot_rewards(ax1, results_dict)
        
        # Bottom plot: % Optimal Action
        ax2 = axes[1]
        self._plot_optimal_actions(ax2, results_dict)
        
        # Add main title
        fig.suptitle('Figure 2.2: Average performance of ε-greedy and gradient bandit algorithms',
                     fontsize=13, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved as '{save_path}'")
        
        return fig
    
    def _plot_rewards(self, ax, results_dict):
        """
        Plot average rewards subplot.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        results_dict : dict
            Experiment results
        """
        for label, results in results_dict.items():
            color = self.colors.get(label, 'black')
            style = self.line_styles.get(label, '-')
            ax.plot(results['avg_rewards'], 
                   label=label, 
                   color=color, 
                   linestyle=style,
                   linewidth=1.5,
                   alpha=0.8)
        
        # Formatting
        ax.set_ylabel('Average\nreward', fontsize=11)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1, linestyle=':')
    
    def _plot_optimal_actions(self, ax, results_dict):
        """
        Plot percentage of optimal actions subplot.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        results_dict : dict
            Experiment results
        """
        for label, results in results_dict.items():
            color = self.colors.get(label, 'black')
            style = self.line_styles.get(label, '-')
            ax.plot(results['pct_optimal'], 
                   label=label, 
                   color=color,
                   linestyle=style,
                   linewidth=1.5,
                   alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel('%\nOptimal\naction', fontsize=11)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1, linestyle=':')
    
    def plot_comparison(self, results_dict, metric='avg_rewards'):
        """
        Create a comparison plot for a specific metric.
        
        Parameters:
        -----------
        results_dict : dict
            Experiment results
        metric : str
            Metric to plot ('avg_rewards' or 'pct_optimal')
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, results in results_dict.items():
            color = self.colors.get(label, 'black')
            style = self.line_styles.get(label, '-')
            ax.plot(results[metric], 
                   label=label, 
                   color=color,
                   linestyle=style,
                   linewidth=2,
                   alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel('Steps', fontsize=12)
        if metric == 'avg_rewards':
            ax.set_ylabel('Average Reward', fontsize=12)
            ax.set_ylim(0, 1.6)
        else:
            ax.set_ylabel('% Optimal Action', fontsize=12)
            ax.set_ylim(0, 100)
        
        ax.set_title(f'Algorithm Comparison: {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, results_dict, window_size=50):
        """
        Plot smoothed learning curves.
        
        Parameters:
        -----------
        results_dict : dict
            Experiment results
        window_size : int
            Window size for moving average smoothing
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, results in results_dict.items():
            # Apply moving average for smoothing
            rewards = results['avg_rewards']
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            color = self.colors.get(label, 'black')
            ax.plot(smoothed, label=label, color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Average Reward (smoothed)', fontsize=12)
        ax.set_title(f'Learning Curves (smoothed with window={window_size})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        return fig