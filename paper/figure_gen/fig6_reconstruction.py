#!/usr/bin/env python
"""
Figure 6: ECG Reconstruction Examples with Wave Decomposition
Shows original ECG, reconstruction, and decomposed P, QRS, T components.
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

from common import setup_style, save_figure, COLORS, OUTPUT_DIR


def generate_figure(model=None, dataloader=None, output_dir: Path = None, device='cuda'):
    """Generate ECG reconstruction examples with wave decomposition (Figure 6)."""
    setup_style()
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("\n[Figure 6] Reconstruction Examples with Wave Decomposition")
    
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
    
    model.eval()
    
    # Get a few examples
    batch = next(iter(dataloader))
    x = batch['x'][:4].to(device)
    
    with torch.no_grad():
        z, _ = model.encoder(x, return_attention=False)
        x_hat, params, components = model.decoder(z, x.shape[-1], return_components=True)
    
    # Larger figure with better proportions
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    
    t = np.arange(x.shape[-1]) / 100  # 100Hz sampling for PTB-XL
    
    for i in range(4):
        # Original + Reconstruction (Lead II, index 1)
        ax_main = axes[i, 0]
        ax_main.plot(t, x[i, 1].cpu().numpy(), color=COLORS['full_ecg'], 
                     linewidth=1.5, label='Original', alpha=0.8)
        ax_main.plot(t, x_hat[i, 1].cpu().numpy(), color=COLORS['ep_prior'], 
                     linewidth=2, label='Reconstruction', linestyle='--')
        ax_main.set_ylabel(f'Sample {i+1}\nAmplitude')
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3, linewidth=0.8)
        if i == 0:
            ax_main.set_title('Original vs Reconstruction (Lead II)')
        if i == 3:
            ax_main.set_xlabel('Time (s)')
        
        # Wave Decomposition
        ax_waves = axes[i, 1]
        ax_waves.plot(t, components['P'][i, 1].cpu().numpy(), color=COLORS['P_wave'],
                     linewidth=2, label='P-wave')
        ax_waves.plot(t, components['QRS'][i, 1].cpu().numpy(), color=COLORS['QRS'],
                     linewidth=2, label='QRS')
        ax_waves.plot(t, components['T'][i, 1].cpu().numpy(), color=COLORS['T_wave'],
                     linewidth=2, label='T-wave')
        ax_waves.legend(loc='upper right')
        ax_waves.grid(True, alpha=0.3, linewidth=0.8)
        if i == 0:
            ax_waves.set_title('Decomposed Wave Components')
        if i == 3:
            ax_waves.set_xlabel('Time (s)')
    
    plt.suptitle('EP-Prior: Interpretable ECG Decomposition', y=0.98)
    plt.tight_layout()
    
    save_figure(fig, 'fig6_reconstruction', output_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()

