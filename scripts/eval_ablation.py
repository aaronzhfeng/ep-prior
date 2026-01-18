#!/usr/bin/env python
"""
Ablation Study Evaluation

Compares three model variants:
1. EP-Prior (full): Structured latent + EP constraints + contrastive
2. EP-Prior (no EP): Structured latent + contrastive (no EP constraints)
3. Baseline: Unstructured latent + contrastive (no EP structure)

This isolates the contributions of:
- Structured latent space (comparing 2 vs 3)
- EP constraints (comparing 1 vs 2)
"""

import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.models.baseline_model import BaselineSSL
from ep_prior.data import PTBXLDataset
from ep_prior.eval.fewshot import FewShotEvaluator


COLORS = {
    'ep_prior_full': '#2E86AB',
    'ep_prior_no_ep': '#7BC950',
    'baseline': '#A23B72',
}

MODEL_NAMES = {
    'ep_prior_full': 'EP-Prior (Full)',
    'ep_prior_no_ep': 'EP-Prior (No EP Loss)',
    'baseline': 'Baseline (Unstructured)',
}


def load_models(checkpoints: dict, device: str):
    """Load all model variants."""
    models = {}
    
    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"  Skipping {name}: checkpoint not found")
            continue
            
        print(f"  Loading {name}...")
        
        if 'baseline' in name:
            model = BaselineSSL.load_from_checkpoint(
                ckpt_path, map_location=device, weights_only=False
            )
        else:
            model = EPPriorSSL.load_from_checkpoint(
                ckpt_path, map_location=device, weights_only=False
            )
        
        model.eval()
        model.to(device)
        models[name] = model
    
    return models


def extract_embeddings(model, dataloader, device, is_structured=True):
    """Extract embeddings from a model."""
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting", leave=False):
            x = batch['x'].to(device)
            
            if is_structured:
                z, _ = model.encoder(x, return_attention=False)
                z_concat = model.encoder.get_latent_concat(z)
            else:
                z_concat = model.encoder(x)
            
            all_z.append(z_concat.cpu())
            
            if 'superclass' in batch:
                all_labels.append(batch['superclass'])
    
    embeddings = torch.cat(all_z, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    
    return embeddings, labels


def fewshot_comparison(models: dict, train_dataset, test_dataset, 
                       device: str, shots: list, n_seeds: int = 3):
    """
    Run few-shot evaluation on all models.
    """
    print("\n" + "="*60)
    print("FEW-SHOT CLASSIFICATION COMPARISON")
    print("="*60)
    
    conditions = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {MODEL_NAMES[model_name]}")
        
        is_structured = 'baseline' not in model_name
        
        # Extract embeddings
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        train_z, train_labels = extract_embeddings(model, train_loader, device, is_structured)
        test_z, test_labels = extract_embeddings(model, test_loader, device, is_structured)
        
        # Few-shot eval
        for n_shot in shots:
            for seed in range(n_seeds):
                np.random.seed(seed)
                
                for cond_idx, condition in enumerate(conditions):
                    # Get positive/negative indices
                    pos_mask = train_labels[:, cond_idx] > 0
                    neg_mask = train_labels[:, cond_idx] == 0
                    
                    pos_idx = np.where(pos_mask.numpy())[0]
                    neg_idx = np.where(neg_mask.numpy())[0]
                    
                    if len(pos_idx) < n_shot or len(neg_idx) < n_shot:
                        continue
                    
                    # Sample
                    selected_pos = np.random.choice(pos_idx, n_shot, replace=False)
                    selected_neg = np.random.choice(neg_idx, n_shot, replace=False)
                    selected = np.concatenate([selected_pos, selected_neg])
                    
                    # Train probe
                    probe_X = train_z[selected].numpy()
                    probe_y = train_labels[selected, cond_idx].numpy()
                    
                    probe = LogisticRegression(max_iter=1000, solver='lbfgs')
                    probe.fit(probe_X, probe_y)
                    
                    # Evaluate
                    test_y = test_labels[:, cond_idx].numpy()
                    test_proba = probe.predict_proba(test_z.numpy())[:, 1]
                    
                    auroc = roc_auc_score(test_y, test_proba)
                    
                    results.append({
                        'model': model_name,
                        'model_display': MODEL_NAMES[model_name],
                        'n_shots': n_shot,
                        'seed': seed,
                        'condition': condition,
                        'auroc': auroc,
                    })
        
        print(f"  Completed {len(shots) * n_seeds * len(conditions)} evaluations")
    
    return pd.DataFrame(results)


def plot_ablation_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Generate ablation comparison plots."""
    
    # Plot 1: Overall comparison by shot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shots = sorted(results_df['n_shots'].unique())
    
    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name]
        means = model_data.groupby('n_shots')['auroc'].mean()
        stds = model_data.groupby('n_shots')['auroc'].std()
        
        ax.errorbar(
            shots, means.values, yerr=stds.values,
            label=MODEL_NAMES[model_name],
            color=COLORS.get(model_name, 'gray'),
            marker='o', linewidth=2, markersize=8, capsize=4
        )
    
    ax.set_xlabel('Labeled Samples per Class', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(shots)
    ax.set_xticklabels(shots)
    ax.set_ylim([0.5, 1.0])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_comparison.pdf')
    fig.savefig(output_dir / 'ablation_comparison.png')
    plt.close(fig)
    print(f"  Saved: ablation_comparison.pdf")
    
    # Plot 2: Bar chart showing delta at specific shot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    target_shot = 50  # Show results at 50-shot
    shot_data = results_df[results_df['n_shots'] == target_shot]
    
    means = shot_data.groupby('model')['auroc'].mean()
    stds = shot_data.groupby('model')['auroc'].std()
    
    models_order = ['baseline', 'ep_prior_no_ep', 'ep_prior_full']
    models_order = [m for m in models_order if m in means.index]
    
    x = np.arange(len(models_order))
    colors = [COLORS[m] for m in models_order]
    
    bars = ax.bar(x, [means[m] for m in models_order], 
                  yerr=[stds[m] for m in models_order],
                  color=colors, capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models_order], fontsize=10)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'Ablation at {target_shot}-shot', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, m in zip(bars, models_order):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{means[m]:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_bar.pdf')
    fig.savefig(output_dir / 'ablation_bar.png')
    plt.close(fig)
    print(f"  Saved: ablation_bar.pdf")


def generate_summary_table(results_df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    
    print("\n" + "-"*60)
    print("ABLATION SUMMARY")
    print("-"*60)
    
    # Compute statistics
    summary = results_df.groupby(['model', 'n_shots']).agg({
        'auroc': ['mean', 'std']
    }).round(3)
    
    summary.columns = ['mean', 'std']
    summary = summary.reset_index()
    
    # Pivot for nice display
    pivot = summary.pivot(index='n_shots', columns='model', values='mean')
    
    print("\nMean AUROC by shot:")
    print(pivot.to_string())
    
    # Compute deltas
    if 'ep_prior_full' in pivot.columns and 'baseline' in pivot.columns:
        print("\n\nDelta (EP-Prior Full vs Baseline):")
        for shot in pivot.index:
            delta = pivot.loc[shot, 'ep_prior_full'] - pivot.loc[shot, 'baseline']
            print(f"  {shot}-shot: {delta:+.3f}")
    
    if 'ep_prior_full' in pivot.columns and 'ep_prior_no_ep' in pivot.columns:
        print("\nDelta (EP-Prior Full vs No EP):")
        for shot in pivot.index:
            delta = pivot.loc[shot, 'ep_prior_full'] - pivot.loc[shot, 'ep_prior_no_ep']
            print(f"  {shot}-shot: {delta:+.3f}")
    
    # Save table
    summary.to_csv(output_dir / 'ablation_summary.csv', index=False)
    print(f"\n  Saved: ablation_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Ablation study evaluation")
    parser.add_argument("--ep_prior_full_ckpt", type=str,
                        default="/root/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt")
    parser.add_argument("--ep_prior_no_ep_ckpt", type=str, default=None,
                        help="Checkpoint for EP-Prior without EP constraints")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="/root/ep-prior/runs/baseline_v1_contrastive/checkpoints/last.ckpt")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--shots", type=str, default="10,50,100,500")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    shots = [int(s) for s in args.shots.split(',')]
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/root/ep-prior/runs/ablation_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Prepare checkpoints
    checkpoints = {
        'ep_prior_full': args.ep_prior_full_ckpt,
        'ep_prior_no_ep': args.ep_prior_no_ep_ckpt,
        'baseline': args.baseline_ckpt,
    }
    
    # Load models
    print("\nLoading models...")
    models = load_models(checkpoints, args.device)
    
    if len(models) < 2:
        print("ERROR: Need at least 2 models for comparison")
        return
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PTBXLDataset(args.data_path, split="train", return_labels=True)
    test_dataset = PTBXLDataset(args.data_path, split="test", return_labels=True)
    
    # Run evaluation
    results_df = fewshot_comparison(
        models, train_dataset, test_dataset,
        args.device, shots, args.n_seeds
    )
    
    # Save results
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'ablation_results.csv'}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_ablation_comparison(results_df, output_dir)
    
    # Generate summary
    generate_summary_table(results_df, output_dir)


if __name__ == "__main__":
    main()

