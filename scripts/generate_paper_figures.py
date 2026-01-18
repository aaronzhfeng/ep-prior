#!/usr/bin/env python
"""
Generate Paper-Quality Figures for EP-Prior

Generates:
1. Sample Efficiency Curves (Main result figure)
2. Intervention Selectivity Heatmap
3. Concept-to-Component Mapping Heatmap
4. Reconstructed ECG Examples with Wave Decomposition
5. Latent Space Visualization (t-SNE)
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.models.baseline_model import BaselineSSL
from ep_prior.data import PTBXLDataset

# Plot style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'ep_prior': '#2E86AB',      # Blue
    'baseline': '#A23B72',       # Magenta  
    'P_wave': '#E8871E',         # Orange
    'QRS': '#1E88E5',            # Blue
    'T_wave': '#43A047',         # Green
    'full_ecg': '#212121',       # Dark gray
}


def figure_sample_efficiency(results_dir: Path, output_dir: Path):
    """
    Generate sample efficiency curves showing AUROC vs # training samples.
    This is the main result figure.
    """
    print("\n[Figure 1] Sample Efficiency Curves")
    
    # Load results
    ep_prior_df = pd.read_csv(results_dir / "fewshot_ep_prior.csv")
    baseline_df = pd.read_csv(results_dir / "fewshot_baseline.csv")
    
    # Get unique shot values and conditions
    shots = sorted(ep_prior_df['n_shots'].unique())
    conditions = ep_prior_df['condition'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, condition in enumerate(conditions[:6]):
        ax = axes[idx]
        
        ep_data = ep_prior_df[ep_prior_df['condition'] == condition]
        base_data = baseline_df[baseline_df['condition'] == condition]
        
        # Group by shots
        ep_means = ep_data.groupby('n_shots')['auroc'].mean()
        ep_stds = ep_data.groupby('n_shots')['auroc'].std()
        base_means = base_data.groupby('n_shots')['auroc'].mean()
        base_stds = base_data.groupby('n_shots')['auroc'].std()
        
        # Plot
        ax.errorbar(ep_means.index, ep_means.values, yerr=ep_stds.values,
                   label='EP-Prior', color=COLORS['ep_prior'], marker='o',
                   linewidth=2, markersize=6, capsize=3)
        ax.errorbar(base_means.index, base_means.values, yerr=base_stds.values,
                   label='Baseline', color=COLORS['baseline'], marker='s',
                   linewidth=2, markersize=6, capsize=3, linestyle='--')
        
        ax.set_xlabel('Labeled samples per class')
        ax.set_ylabel('AUROC')
        ax.set_title(condition)
        ax.set_xscale('log')
        ax.set_xticks(shots)
        ax.set_xticklabels(shots)
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    # Remove empty subplots
    for idx in range(len(conditions), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sample Efficiency: EP-Prior vs Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig1_sample_efficiency.pdf')
    fig.savefig(output_dir / 'fig1_sample_efficiency.png')
    plt.close(fig)
    print(f"  Saved: fig1_sample_efficiency.pdf")


def figure_intervention_heatmap(model, dataloader, output_dir: Path, device='cuda'):
    """
    Generate intervention selectivity heatmap showing which latents affect which params.
    """
    print("\n[Figure 2] Intervention Selectivity Heatmap")
    
    from ep_prior.eval.intervention import InterventionTester
    
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
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    im = ax.imshow(effect_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(6))
    ax.set_xticklabels(['P τ', 'P σ', 'QRS τ', 'QRS σ', 'T τ', 'T σ'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Vary z_P', 'Vary z_QRS', 'Vary z_T'])
    ax.set_xlabel('Decoder Parameters Changed')
    ax.set_ylabel('Latent Intervention')
    ax.set_title('Intervention Selectivity: Disentanglement Verification', fontweight='bold')
    
    # Add value annotations
    for i in range(3):
        for j in range(6):
            val = effect_matrix_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    # Highlight diagonal blocks (expected high values)
    for i in range(3):
        rect = plt.Rectangle((i*2-0.5, i-0.5), 2, 1, fill=False, 
                              edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
    
    plt.colorbar(im, ax=ax, label='Normalized Effect')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig2_intervention_heatmap.pdf')
    fig.savefig(output_dir / 'fig2_intervention_heatmap.png')
    plt.close(fig)
    print(f"  Saved: fig2_intervention_heatmap.pdf")


def figure_concept_mapping(results_dir: Path, output_dir: Path):
    """
    Generate concept-to-component mapping heatmap.
    Shows which latent components predict which clinical concepts.
    """
    print("\n[Figure 3] Concept-to-Component Mapping")
    
    # Load concept predictability results
    concept_file = results_dir / "concept_predictability.csv"
    if not concept_file.exists():
        print("  Skipping: concept_predictability.csv not found")
        return
    
    df = pd.read_csv(concept_file)
    
    # Create matrix
    components = ['z_P', 'z_QRS', 'z_T', 'z_HRV', 'z_all']
    concepts = df['concept'].unique()
    
    matrix = np.zeros((len(concepts), len(components)))
    
    for i, concept in enumerate(concepts):
        for j, comp in enumerate(components):
            row = df[(df['concept'] == concept) & (df['component'] == comp)]
            if len(row) > 0:
                matrix[i, j] = row['auroc'].values[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(['z_P\n(P-wave)', 'z_QRS\n(QRS)', 'z_T\n(T-wave)', 
                        'z_HRV\n(Rhythm)', 'z_all\n(Full)'])
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    ax.set_xlabel('Latent Component')
    ax.set_ylabel('Clinical Concept')
    ax.set_title('Concept Predictability from Structured Latents', fontweight='bold')
    
    # Add value annotations
    for i in range(len(concepts)):
        for j in range(len(components)):
            val = matrix[i, j]
            color = 'white' if val > 0.75 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    # Highlight expected high values
    expected = {
        'STTC': 2,  # T-wave
        'CD': 1,    # QRS
    }
    for concept, comp_idx in expected.items():
        if concept in concepts:
            i = list(concepts).index(concept)
            rect = plt.Rectangle((comp_idx-0.5, i-0.5), 1, 1, fill=False,
                                 edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
    
    plt.colorbar(im, ax=ax, label='AUROC')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig3_concept_mapping.pdf')
    fig.savefig(output_dir / 'fig3_concept_mapping.png')
    plt.close(fig)
    print(f"  Saved: fig3_concept_mapping.pdf")


def figure_reconstruction_examples(model, dataloader, output_dir: Path, device='cuda'):
    """
    Generate ECG reconstruction examples with wave decomposition.
    Shows original, reconstruction, and individual wave components.
    """
    print("\n[Figure 4] Reconstruction Examples with Wave Decomposition")
    
    model.eval()
    
    # Get a few examples
    batch = next(iter(dataloader))
    x = batch['x'][:4].to(device)
    
    with torch.no_grad():
        z, _ = model.encoder(x, return_attention=False)
        x_hat, params = model.decoder(z, x.shape[-1], return_components=True)
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    t = np.arange(x.shape[-1]) / 500  # Assuming 500Hz sampling
    
    for i in range(4):
        # Original + Reconstruction (Lead II, index 1)
        ax_main = axes[i, 0]
        ax_main.plot(t, x[i, 1].cpu().numpy(), color=COLORS['full_ecg'], 
                     linewidth=1, label='Original', alpha=0.7)
        ax_main.plot(t, x_hat[i, 1].cpu().numpy(), color=COLORS['ep_prior'], 
                     linewidth=1.5, label='Reconstruction', linestyle='--')
        ax_main.set_ylabel(f'Sample {i+1}\nAmplitude (mV)')
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)
        if i == 0:
            ax_main.set_title('Original vs Reconstruction (Lead II)', fontweight='bold')
        if i == 3:
            ax_main.set_xlabel('Time (s)')
        
        # Wave Decomposition
        ax_waves = axes[i, 1]
        if 'components' in params:
            comp = params['components']
            ax_waves.plot(t, comp['P'][i, 1].cpu().numpy(), color=COLORS['P_wave'],
                         linewidth=1.5, label='P-wave')
            ax_waves.plot(t, comp['QRS'][i, 1].cpu().numpy(), color=COLORS['QRS'],
                         linewidth=1.5, label='QRS')
            ax_waves.plot(t, comp['T'][i, 1].cpu().numpy(), color=COLORS['T_wave'],
                         linewidth=1.5, label='T-wave')
        ax_waves.legend(loc='upper right')
        ax_waves.grid(True, alpha=0.3)
        if i == 0:
            ax_waves.set_title('Decomposed Wave Components', fontweight='bold')
        if i == 3:
            ax_waves.set_xlabel('Time (s)')
    
    plt.suptitle('EP-Prior: Interpretable ECG Decomposition', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig4_reconstruction_examples.pdf')
    fig.savefig(output_dir / 'fig4_reconstruction_examples.png')
    plt.close(fig)
    print(f"  Saved: fig4_reconstruction_examples.pdf")


def figure_latent_tsne(model, dataloader, output_dir: Path, device='cuda', n_samples=1000):
    """
    Generate t-SNE visualization of latent space colored by pathology.
    """
    print("\n[Figure 5] Latent Space t-SNE Visualization")
    
    from sklearn.manifold import TSNE
    
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
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(all_z)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: colored by dominant pathology
    ax1 = axes[0]
    if len(all_labels) > 0:
        superclass_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        colors_list = plt.cm.tab10.colors[:5]
        
        dominant_class = all_labels.argmax(dim=1).numpy()
        
        for i, name in enumerate(superclass_names):
            mask = dominant_class == i
            ax1.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors_list[i]], 
                       label=name, alpha=0.6, s=10)
        
        ax1.legend(title='Dominant Class', loc='best')
    else:
        ax1.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, s=10)
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Latent Space (by Pathology)', fontweight='bold')
    
    # Right: individual components
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
                   color=[COLORS['P_wave'], COLORS['QRS'], COLORS['T_wave'], '#9E9E9E'])
    ax2.set_ylabel('Total Variance')
    ax2.set_title('Information Distribution Across Components', fontweight='bold')
    
    for bar, val in zip(bars, variances.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('EP-Prior: Structured Latent Space', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig5_latent_tsne.pdf')
    fig.savefig(output_dir / 'fig5_latent_tsne.png')
    plt.close(fig)
    print(f"  Saved: fig5_latent_tsne.pdf")


def figure_comparison_table(results_dir: Path, output_dir: Path):
    """
    Generate a nice comparison table as a figure.
    """
    print("\n[Table 1] Main Results Comparison")
    
    # Aggregate results
    ep_df = pd.read_csv(results_dir / "fewshot_ep_prior.csv")
    base_df = pd.read_csv(results_dir / "fewshot_baseline.csv")
    
    # Compute mean AUROC per shot
    shots = sorted(ep_df['n_shots'].unique())
    
    table_data = []
    for shot in shots:
        ep_mean = ep_df[ep_df['n_shots'] == shot]['auroc'].mean()
        ep_std = ep_df[ep_df['n_shots'] == shot]['auroc'].std()
        base_mean = base_df[base_df['n_shots'] == shot]['auroc'].mean()
        base_std = base_df[base_df['n_shots'] == shot]['auroc'].std()
        delta = ep_mean - base_mean
        
        table_data.append({
            'Shots': shot,
            'EP-Prior': f'{ep_mean:.3f} ± {ep_std:.3f}',
            'Baseline': f'{base_mean:.3f} ± {base_std:.3f}',
            'Δ': f'{delta:+.3f}',
        })
    
    df = pd.DataFrame(table_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 4,
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Few-Shot Classification: Mean AUROC Across All Conditions', 
                 fontweight='bold', fontsize=13, pad=20)
    
    fig.savefig(output_dir / 'table1_comparison.pdf')
    fig.savefig(output_dir / 'table1_comparison.png')
    plt.close(fig)
    print(f"  Saved: table1_comparison.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality figures")
    parser.add_argument("--results_dir", type=str, 
                        default="/root/ep-prior/runs/evaluation_01")
    parser.add_argument("--ep_prior_ckpt", type=str,
                        default="/root/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/root/ep-prior/runs/figures_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    print("\nLoading EP-Prior model...")
    model = EPPriorSSL.load_from_checkpoint(
        args.ep_prior_ckpt, map_location=args.device, weights_only=False
    )
    model.eval()
    model.to(args.device)
    
    # Load data
    print("Loading test dataset...")
    test_dataset = PTBXLDataset(args.data_path, split='test', return_labels=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Generate figures
    print("\n" + "="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)
    
    figure_sample_efficiency(results_dir, output_dir)
    figure_intervention_heatmap(model, test_loader, output_dir, args.device)
    figure_concept_mapping(results_dir, output_dir)
    figure_reconstruction_examples(model, test_loader, output_dir, args.device)
    figure_latent_tsne(model, test_loader, output_dir, args.device)
    figure_comparison_table(results_dir, output_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

