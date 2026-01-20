#!/usr/bin/env python
"""
Figure 3: Intervention Selectivity Heatmap
Shows which latent components control which decoder parameters (disentanglement).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/root/workspace/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataset
from ep_prior.eval.intervention import InterventionTester

from common import setup_style, save_figure, COLORS, OUTPUT_DIR, CELL_TEXT_SIZE, COLORBAR_LABEL_SIZE


def generate_figure(model=None, dataloader=None, output_dir: Path = None, device='cuda'):
    """Generate intervention selectivity heatmap (Figure 3)."""
    setup_style()
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("\n[Figure 3] Intervention Selectivity Heatmap")
    
    # Load model if not provided
    if model is None:
        checkpoint_path = "/root/workspace/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt"
        model = EPPriorSSL.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False)
        model.eval()
        model.to(device)
    
    # Load data if not provided
    if dataloader is None:
        test_dataset = PTBXLDataset("/root/workspace/ep-prior/data/ptb-xl", split='test', return_labels=True)
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    tester = InterventionTester(model, device=device)
    baseline_z = tester._get_baseline_latents(dataloader, n_samples=50)
    
    # Compute intervention effects matrix
    components = ['P', 'QRS', 'T']
    params = ['tau', 'sig']
    
    effect_matrix = np.zeros((len(components), len(components) * len(params)))
    
    for i, target in enumerate(components):
        direction = tester._compute_principal_direction(baseline_z[target])
        base_z = {k: v[0:1] for k, v in baseline_z.items()}
        results = tester.run_intervention(base_z, target, direction, n_steps=11, scale=2.0)
        
        for j, wave in enumerate(components):
            for k, param in enumerate(params):
                vals = results['params'][wave][param]
                effect = (vals.max() - vals.min()).item()
                effect_matrix[i, j*2 + k] = effect
    
    # Normalize by row max for visualization
    row_max = effect_matrix.max(axis=1, keepdims=True)
    effect_matrix_norm = effect_matrix / (row_max + 1e-8)
    
    # Larger figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(effect_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Labels with larger text
    ax.set_xticks(range(6))
    ax.set_xticklabels(['P τ', 'P σ', 'QRS τ', 'QRS σ', 'T τ', 'T σ'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Vary z_P', 'Vary z_QRS', 'Vary z_T'])
    ax.set_xlabel('Decoder Parameters Changed')
    ax.set_ylabel('Latent Intervention')
    ax.set_title('Intervention Selectivity: Disentanglement Verification')
    
    # Add value annotations with larger font
    for i in range(3):
        for j in range(6):
            val = effect_matrix_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=CELL_TEXT_SIZE, fontweight='bold')
    
    # Highlight diagonal blocks (expected high values)
    for i in range(3):
        rect = plt.Rectangle((i*2-0.5, i-0.5), 2, 1, fill=False, 
                              edgecolor='red', linewidth=2.5, linestyle='--')
        ax.add_patch(rect)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Effect', fontsize=COLORBAR_LABEL_SIZE)
    plt.tight_layout()
    
    save_figure(fig, 'fig3_intervention', output_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()

