#!/usr/bin/env python3
"""
Generate thesis-quality figures from experiment results.

Produces:
  1. Learning trajectories (mAP@50 vs cumulative labels) per L0
  2. AUC bar chart comparison
  3. Final metrics comparison table
  4. Latency analysis
  5. Per-cycle precision/recall evolution

Usage:
    python scripts/plot_results.py --results results/all_results_*.json --output figuras/
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# Thesis-consistent styling
STRATEGY_STYLES = {
    'random':          {'color': '#95A5A6', 'marker': 's', 'linestyle': '--',  'label': 'Random Sampling'},
    'deterministic':   {'color': '#E67E22', 'marker': '^', 'linestyle': '-.',  'label': 'Deterministic Uncertainty'},
    'bald_only':       {'color': '#9B59B6', 'marker': 'D', 'linestyle': ':',   'label': 'BALD-only (MC Dropout)'},
    'bald_diversity':  {'color': '#3498DB', 'marker': 'o', 'linestyle': '-',   'label': 'BALD + Diversity (Proposed)'},
}


def load_results(results_path):
    """Load aggregated results JSON."""
    with open(results_path) as f:
        return json.load(f)


def extract_trajectories(results):
    """
    Organize results by L0 size and strategy.
    Returns: dict[L0][strategy] -> trajectory list
    """
    organized = {}
    
    for key, data in results.items():
        if 'experiment' not in data:
            continue
        
        exp = data['experiment']
        L0 = exp['L0_size']
        strategy = exp['strategy']
        
        if L0 not in organized:
            organized[L0] = {}
        
        organized[L0][strategy] = data
    
    return organized


def plot_learning_trajectories(organized, output_dir):
    """
    Plot mAP@50 vs cumulative labels for each L0 regime.
    One figure per L0 size.
    """
    for L0, strategies in sorted(organized.items()):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        for strategy, data in strategies.items():
            if strategy not in STRATEGY_STYLES:
                continue
            
            style = STRATEGY_STYLES[strategy]
            traj = data.get('trajectory', [])
            
            if not traj:
                continue
            
            x = [t['cumulative_labels'] for t in traj]
            y = [t['mAP50'] for t in traj]
            
            ax.plot(x, y, 
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    markersize=7,
                    label=style['label'])
        
        ax.set_xlabel('Cumulative Labelled Samples', fontsize=12)
        ax.set_ylabel('mAP@50', fontsize=12)
        ax.set_title(f'Learning Trajectory ($L_0 = {L0}$)', fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        filepath = output_dir / f'trajectory_L0_{L0}.pdf'
        plt.savefig(filepath, bbox_inches='tight')
        plt.savefig(filepath.with_suffix('.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")


def plot_auc_comparison(organized, output_dir):
    """
    Bar chart comparing AUC across strategies for each L0.
    """
    L0_sizes = sorted(organized.keys())
    strategies = list(STRATEGY_STYLES.keys())
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    x = np.arange(len(L0_sizes))
    width = 0.18
    
    for i, strategy in enumerate(strategies):
        if strategy not in STRATEGY_STYLES:
            continue
        
        style = STRATEGY_STYLES[strategy]
        aucs = []
        
        for L0 in L0_sizes:
            if L0 in organized and strategy in organized[L0]:
                aucs.append(organized[L0][strategy].get('auc', 0))
            else:
                aucs.append(0)
        
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, 
                       color=style['color'], 
                       label=style['label'],
                       alpha=0.85,
                       edgecolor='white',
                       linewidth=0.5)
    
    ax.set_xlabel('Initial Labelled Set Size ($L_0$)', fontsize=12)
    ax.set_ylabel('Area Under the Learning Curve (AUC)', fontsize=12)
    ax.set_title('Annotation Efficiency Comparison (H1)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L0) for L0 in L0_sizes])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    filepath = output_dir / 'auc_comparison.pdf'
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_final_metrics(organized, output_dir):
    """
    Grouped bar chart of final mAP@50, Precision, Recall for each strategy.
    Uses the default L0 size (first one).
    """
    L0 = sorted(organized.keys())[0]
    strategies_data = organized[L0]
    
    strategies = [s for s in STRATEGY_STYLES.keys() if s in strategies_data]
    
    metrics_names = ['mAP@50', 'Precision', 'Recall']
    metrics_keys = ['final_mAP50', 'final_precision', 'final_recall']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    x = np.arange(len(metrics_names))
    width = 0.18
    
    for i, strategy in enumerate(strategies):
        style = STRATEGY_STYLES[strategy]
        data = strategies_data[strategy]
        
        values = [data.get(k, 0) for k in metrics_keys]
        
        offset = (i - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, values, width,
               color=style['color'],
               label=style['label'],
               alpha=0.85,
               edgecolor='white')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Final Detection Metrics ($L_0 = {L0}$)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    filepath = output_dir / 'final_metrics.pdf'
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_precision_recall_evolution(organized, output_dir):
    """
    Plot precision and recall evolution across cycles.
    """
    L0 = sorted(organized.keys())[0]
    strategies_data = organized[L0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for strategy, data in strategies_data.items():
        if strategy not in STRATEGY_STYLES:
            continue
        
        style = STRATEGY_STYLES[strategy]
        traj = data.get('trajectory', [])
        
        if not traj:
            continue
        
        cycles = [t['cycle'] for t in traj]
        precision = [t['precision'] for t in traj]
        recall = [t['recall'] for t in traj]
        
        ax1.plot(cycles, precision,
                 color=style['color'], marker=style['marker'],
                 linestyle=style['linestyle'], linewidth=2,
                 markersize=6, label=style['label'])
        
        ax2.plot(cycles, recall,
                 color=style['color'], marker=style['marker'],
                 linestyle=style['linestyle'], linewidth=2,
                 markersize=6, label=style['label'])
    
    ax1.set_xlabel('Adaptation Cycle', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision Evolution', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Adaptation Cycle', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Recall Evolution (H2)', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'precision_recall_evolution.pdf'
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_latency_analysis(organized, output_dir):
    """
    Bar chart of inference latency per strategy.
    """
    L0 = sorted(organized.keys())[0]
    strategies_data = organized[L0]
    
    strategies = [s for s in STRATEGY_STYLES.keys() if s in strategies_data]
    latencies = [strategies_data[s].get('mean_latency_ms', 0) for s in strategies]
    colors = [STRATEGY_STYLES[s]['color'] for s in strategies]
    labels = [STRATEGY_STYLES[s]['label'] for s in strategies]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    bars = ax.barh(range(len(strategies)), latencies, color=colors, alpha=0.85,
                    edgecolor='white', height=0.6)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean Inference Latency (ms)', fontsize=12)
    ax.set_title('Operational Feasibility (H4)', fontsize=13)
    ax.grid(True, alpha=0.2, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, latencies)):
        ax.text(val + 0.3, i, f'{val:.1f} ms', va='center', fontsize=10)
    
    plt.tight_layout()
    filepath = output_dir / 'latency_analysis.pdf'
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_latex_table(organized, output_dir):
    """Generate LaTeX table for inclusion in the thesis."""
    lines = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of acquisition strategies across prior-knowledge regimes.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{$L_0$} & \textbf{Strategy} & \textbf{mAP@50} & \textbf{Precision} & \textbf{Recall} & \textbf{AUC} & \textbf{Latency (ms)} \\")
    lines.append(r"\hline")
    
    for L0 in sorted(organized.keys()):
        first = True
        for strategy in STRATEGY_STYLES.keys():
            if strategy not in organized[L0]:
                continue
            
            data = organized[L0][strategy]
            label = STRATEGY_STYLES[strategy]['label']
            
            L0_str = str(L0) if first else ""
            first = False
            
            lines.append(
                f"{L0_str} & {label} & "
                f"{data.get('final_mAP50', 0):.4f} & "
                f"{data.get('final_precision', 0):.4f} & "
                f"{data.get('final_recall', 0):.4f} & "
                f"{data.get('auc', 0):.1f} & "
                f"{data.get('mean_latency_ms', 0):.1f} \\\\"
            )
        lines.append(r"\hline")
    
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    tex_path = output_dir / "results_table.tex"
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures from results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to aggregated results JSON")
    parser.add_argument("--output", type=str, default="figuras",
                       help="Output directory for figures")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(args.results)
    organized = extract_trajectories(results)
    
    print(f"Found data for L0 sizes: {sorted(organized.keys())}")
    
    print("\nGenerating figures...")
    plot_learning_trajectories(organized, output_dir)
    plot_auc_comparison(organized, output_dir)
    plot_final_metrics(organized, output_dir)
    plot_precision_recall_evolution(organized, output_dir)
    plot_latency_analysis(organized, output_dir)
    generate_latex_table(organized, output_dir)
    
    print("\nDone! All figures saved to:", output_dir)


if __name__ == "__main__":
    main()
