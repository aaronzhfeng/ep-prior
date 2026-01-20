#!/usr/bin/env python
"""
Figure 2: Sample Efficiency Curves
Shows AUROC vs # training samples for EP-Prior vs Baseline.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from common import setup_style, save_figure, COLORS, RESULTS_DIR, OUTPUT_DIR, ANNOTATION_SIZE


def generate_figure(results_dir: Path = None, output_dir: Path = None):
    """Generate sample efficiency curves (Figure 2)."""
    setup_style()
    
    if results_dir is None:
        results_dir = RESULTS_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("\n[Figure 2] Sample Efficiency Curves")
    
    # Load results
    ep_prior_df = pd.read_csv(results_dir / "fewshot_ep_prior.csv")
    baseline_df = pd.read_csv(results_dir / "fewshot_baseline.csv")
    
    # Get unique shot values
    shots = sorted(ep_prior_df['shot_size'].unique())
    
    # Filter to concat embedding for main comparison
    ep_concat = ep_prior_df[ep_prior_df['embedding'] == 'concat']
    base_concat = baseline_df[baseline_df['embedding'] == 'concat']
    
    # Larger figure for better readability
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Group by shot_size
    ep_means = ep_concat.groupby('shot_size')['auroc_macro'].mean()
    ep_stds = ep_concat.groupby('shot_size')['auroc_macro'].std()
    base_means = base_concat.groupby('shot_size')['auroc_macro'].mean()
    base_stds = base_concat.groupby('shot_size')['auroc_macro'].std()
    
    # Plot with larger markers and thicker lines
    ax.errorbar(ep_means.index, ep_means.values, yerr=ep_stds.values,
               label='EP-Prior', color=COLORS['ep_prior'], marker='o',
               linewidth=2.5, markersize=10, capsize=5, capthick=2)
    ax.errorbar(base_means.index, base_means.values, yerr=base_stds.values,
               label='Baseline', color=COLORS['baseline'], marker='s',
               linewidth=2.5, markersize=10, capsize=5, capthick=2, linestyle='--')
    
    ax.set_xlabel('Labeled samples per class (k-shot)')
    ax.set_ylabel('Macro AUROC')
    ax.set_title('Sample Efficiency: EP-Prior vs Capacity-Matched Baseline')
    ax.set_xscale('log')
    ax.set_xticks(shots)
    ax.set_xticklabels(shots)
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='lower right')
    
    # Add delta annotation with larger font
    for shot in shots:
        ep_val = ep_means.get(shot, 0)
        base_val = base_means.get(shot, 0)
        delta = ep_val - base_val
        if delta > 0:
            ax.annotate(f'+{delta:.1%}', xy=(shot, ep_val), xytext=(0, 12),
                       textcoords='offset points', ha='center', 
                       fontsize=ANNOTATION_SIZE, fontweight='bold',
                       color=COLORS['ep_prior'])
    
    plt.tight_layout()
    save_figure(fig, 'fig2_sample_efficiency', output_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()

