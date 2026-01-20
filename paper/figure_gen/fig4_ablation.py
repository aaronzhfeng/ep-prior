#!/usr/bin/env python
"""
Figure 4: Ablation Study Bar Chart
Shows impact of removing EP constraints on model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common import setup_style, save_figure, COLORS, RESULTS_DIR, OUTPUT_DIR, ANNOTATION_SIZE


def generate_figure(results_dir: Path = None, output_dir: Path = None):
    """Generate ablation study bar chart (Figure 4)."""
    setup_style()
    
    if results_dir is None:
        results_dir = RESULTS_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("\n[Figure 4] Ablation Study")
    
    # Load ablation results
    ablation_file = results_dir / "ablation_summary.csv"
    if not ablation_file.exists():
        print("  Skipping: ablation_summary.csv not found")
        return
    
    df = pd.read_csv(ablation_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data - use 100-shot for clear comparison
    # CSV structure: model, n_shots, mean, std
    df_100 = df[df['n_shots'] == 100]
    
    # Get values for each model at 100-shot
    models = ['EP-Prior\n(Full)', 'No EP\nConstraint', 'Baseline']
    model_keys = ['ep_prior_full', 'ep_prior_no_ep', 'baseline']
    
    values = []
    stds = []
    for model_key in model_keys:
        row = df_100[df_100['model'] == model_key]
        if len(row) > 0:
            values.append(row['mean'].values[0])
            stds.append(row['std'].values[0])
        else:
            values.append(0.5)
            stds.append(0.0)
    
    # Define colors (EP-Prior blue, failure red, baseline gray)
    bar_colors = [COLORS['ep_prior'], '#E57373', COLORS['baseline']]
    
    # Create bars
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=bar_colors, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('AUROC (100-shot)')
    ax.set_title('Ablation Study: Impact of EP Constraints')
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=ANNOTATION_SIZE, fontweight='bold')
    
    # Add horizontal line for reference
    ax.axhline(y=values[0], color=COLORS['ep_prior'], linestyle='--', alpha=0.5, linewidth=2)
    
    # Highlight the catastrophic failure (No EP constraint)
    if len(values) > 1 and values[1] < 0.7:
        ax.annotate('EP constraints\nessential!',
                   xy=(1, values[1]),
                   xytext=(1.5, 0.72),
                   fontsize=ANNOTATION_SIZE, color='red', fontweight='bold',
                   ha='center',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    save_figure(fig, 'fig4_ablation', output_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()

