#!/usr/bin/env python
"""
Figure 5: Latent Space Visualization (t-SNE)
Shows structured latent space colored by pathology with VERTICAL layout for better readability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, "/root/workspace/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataset

from common import (
    setup_style, save_figure, COLORS, OUTPUT_DIR, 
    ANNOTATION_SIZE, SUPERCLASS_COLORS
)


def generate_figure(model=None, dataloader=None, output_dir: Path = None, device='cuda', n_samples=1000):
    """Generate t-SNE visualization with VERTICAL layout (Figure 5)."""
    setup_style()
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print("\n[Figure 5] Latent Space t-SNE Visualization (Vertical Layout)")
    
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
    
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            x = batch['x'].to(device)
            z, _ = model.encoder(x, return_attention=False)
            z_concat = model.encoder.get_latent_concat(z)
            all_z.append(z_concat.cpu())
            if 'superclass' in batch:
                all_labels.append(batch['superclass'])
            
            if len(all_z) * dataloader.batch_size >= n_samples:
                break
    
    all_z = torch.cat(all_z, dim=0)[:n_samples].numpy()
    
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)[:n_samples]
    
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(all_z)
    
    # VERTICAL LAYOUT: 2 rows, 1 column for better readability
    fig, axes = plt.subplots(2, 1, figsize=(8, 14))
    
    # Top: colored by dominant pathology
    ax1 = axes[0]
    if len(all_labels) > 0:
        superclass_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        colors_list = list(SUPERCLASS_COLORS.values())
        
        dominant_class = all_labels.argmax(dim=1).numpy()
        
        for i, name in enumerate(superclass_names):
            mask = dominant_class == i
            ax1.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors_list[i]], 
                       label=name, alpha=0.7, s=25)  # Larger markers
        
        ax1.legend(title='Dominant Class', loc='best', markerscale=1.5)
    else:
        ax1.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, s=25)
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Latent Space Colored by Pathology')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: information distribution bar chart
    ax2 = axes[1]
    
    # Extract component-specific latents
    d_per_comp = all_z.shape[1] // 4
    z_components = {
        'z_P': all_z[:, :d_per_comp],
        'z_QRS': all_z[:, d_per_comp:2*d_per_comp],
        'z_T': all_z[:, 2*d_per_comp:3*d_per_comp],
        'z_HRV': all_z[:, 3*d_per_comp:],
    }
    
    # Show variance per component
    variances = {k: np.var(v, axis=0).sum() for k, v in z_components.items()}
    
    bars = ax2.bar(variances.keys(), variances.values(), 
                   color=[COLORS['P_wave'], COLORS['QRS'], COLORS['T_wave'], '#9E9E9E'],
                   edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Total Variance')
    ax2.set_xlabel('Latent Component')
    ax2.set_title('Information Distribution Across Components')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, variances.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', 
                fontsize=ANNOTATION_SIZE, fontweight='bold')
    
    plt.suptitle('EP-Prior: Structured Latent Space', y=0.98)
    plt.tight_layout()
    
    save_figure(fig, 'fig5_latent_tsne', output_dir)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()

