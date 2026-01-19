#!/usr/bin/env python
"""
Failure-Mode Stratification by Rhythm Type

Evaluates EP-Prior vs Baseline performance stratified by:
- EP-valid rhythms (normal sinus): expect strong EP-Prior gains
- EP-violated rhythms (AFib, LBBB, WPW): expect graceful degradation

This tests the hypothesis that EP constraints help when they match
the data, and don't hurt when violated.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.models.baseline_model import BaselineSSL
from ep_prior.data import PTBXLDataset


# PTB-XL rhythm stratification groups
RHYTHM_GROUPS = {
    "normal_sinus": {
        "description": "Normal sinus rhythm - EP constraints should be valid",
        "superclass": "NORM",
        "expect_ep_gain": True,
    },
    "conduction_defects": {
        "description": "Bundle branch blocks - wide QRS, EP duration bounds violated",
        "superclass": "CD",
        "expect_ep_gain": False,  # QRS duration constraints may be violated
    },
    "st_t_changes": {
        "description": "ST/T wave changes - T wave morphology altered",
        "superclass": "STTC",
        "expect_ep_gain": True,  # EP constraints still valid for timing
    },
    "hypertrophy": {
        "description": "Ventricular hypertrophy - altered QRS morphology",
        "superclass": "HYP",
        "expect_ep_gain": True,  # Timing still valid
    },
    "myocardial_infarction": {
        "description": "MI patterns - altered morphology",
        "superclass": "MI",
        "expect_ep_gain": True,  # Timing still valid
    },
}


def load_models(ep_prior_ckpt: str, baseline_ckpt: str, device: str):
    """Load both models."""
    ep_prior = EPPriorSSL.load_from_checkpoint(
        ep_prior_ckpt, map_location=device, weights_only=False
    )
    ep_prior.eval()
    ep_prior.to(device)
    
    baseline = BaselineSSL.load_from_checkpoint(
        baseline_ckpt, map_location=device, weights_only=False
    )
    baseline.eval()
    baseline.to(device)
    
    return ep_prior, baseline


def extract_embeddings_for_subset(model, dataset, indices, device, is_structured=True):
    """Extract embeddings for a subset of data."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
    
    all_z = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            
            if is_structured:
                z, _ = model.encoder(x, return_attention=False)
                z_concat = model.encoder.get_latent_concat(z)
            else:
                z_concat = model.encoder(x)
            
            all_z.append(z_concat.cpu())
    
    return torch.cat(all_z, dim=0)


def stratified_evaluation(
    ep_prior, baseline, 
    train_dataset, test_dataset,
    device="cuda",
    n_train_per_class=100,
):
    """
    Evaluate both models stratified by rhythm type.
    
    For each rhythm group:
    1. Get test samples belonging to that group
    2. Train linear probes on full training data
    3. Evaluate on the stratified test subset
    """
    print("\n" + "="*60)
    print("FAILURE-MODE STRATIFICATION ANALYSIS")
    print("="*60)
    
    # Get superclass labels for test set
    test_labels = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if "superclass" in sample:
            test_labels.append(sample["superclass"])
    test_labels = torch.stack(test_labels)
    
    # Map superclass indices
    superclass_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    # Extract all embeddings
    print("\nExtracting embeddings...")
    
    # Training embeddings (for probe training)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    train_z_ep = []
    train_z_base = []
    train_labels = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Train embeddings"):
            x = batch["x"].to(device)
            
            # EP-Prior
            z, _ = ep_prior.encoder(x, return_attention=False)
            train_z_ep.append(ep_prior.encoder.get_latent_concat(z).cpu())
            
            # Baseline
            train_z_base.append(baseline.encoder(x).cpu())
            
            if "superclass" in batch:
                train_labels.append(batch["superclass"])
    
    train_z_ep = torch.cat(train_z_ep, dim=0)
    train_z_base = torch.cat(train_z_base, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    # Test embeddings
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    test_z_ep = []
    test_z_base = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test embeddings"):
            x = batch["x"].to(device)
            
            z, _ = ep_prior.encoder(x, return_attention=False)
            test_z_ep.append(ep_prior.encoder.get_latent_concat(z).cpu())
            test_z_base.append(baseline.encoder(x).cpu())
    
    test_z_ep = torch.cat(test_z_ep, dim=0)
    test_z_base = torch.cat(test_z_base, dim=0)
    
    # Train linear probes (multi-label: one classifier per class)
    print("\nTraining linear probes...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    
    probe_ep = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    probe_ep.fit(train_z_ep.numpy(), train_labels.numpy())
    
    probe_base = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    probe_base.fit(train_z_base.numpy(), train_labels.numpy())
    
    # Evaluate per rhythm group
    results = []
    
    print("\n" + "-"*60)
    print("Results by Rhythm Type")
    print("-"*60)
    
    for group_name, group_info in RHYTHM_GROUPS.items():
        superclass = group_info["superclass"]
        superclass_idx = superclass_names.index(superclass)
        
        # Get indices where this superclass is positive
        group_mask = test_labels[:, superclass_idx] > 0
        n_samples = group_mask.sum().item()
        
        if n_samples < 10:
            print(f"\n{group_name}: Skipping (only {n_samples} samples)")
            continue
        
        # Get predictions for this subset
        test_z_ep_subset = test_z_ep[group_mask]
        test_z_base_subset = test_z_base[group_mask]
        test_labels_subset = test_labels[group_mask]
        
        # Compute AUROC for this superclass (binary: has this condition or not)
        # MultiOutputClassifier.predict_proba returns list of arrays, one per class
        # Each array is (n_samples, 2) for binary; we want probability of class 1
        y_true_full = test_labels[:, superclass_idx].numpy()
        
        # Get probabilities for this superclass (index into list, then get prob of positive class)
        proba_ep_full = probe_ep.predict_proba(test_z_ep.numpy())[superclass_idx][:, 1]
        proba_base_full = probe_base.predict_proba(test_z_base.numpy())[superclass_idx][:, 1]
        
        auroc_ep = roc_auc_score(y_true_full, proba_ep_full)
        auroc_base = roc_auc_score(y_true_full, proba_base_full)
        
        delta = auroc_ep - auroc_base
        expected = "✓" if (delta > 0) == group_info["expect_ep_gain"] else "✗"
        
        results.append({
            "group": group_name,
            "superclass": superclass,
            "n_samples": n_samples,
            "auroc_ep_prior": auroc_ep,
            "auroc_baseline": auroc_base,
            "delta": delta,
            "expected_gain": group_info["expect_ep_gain"],
            "matches_expectation": expected,
        })
        
        print(f"\n{group_name} ({superclass}, n={n_samples}):")
        print(f"  EP-Prior: {auroc_ep:.3f}")
        print(f"  Baseline: {auroc_base:.3f}")
        print(f"  Delta: {delta:+.3f} {'(EP wins)' if delta > 0 else '(Baseline wins)'}")
        print(f"  Expected EP gain: {group_info['expect_ep_gain']} {expected}")
    
    # Summary table
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Failure-mode stratification analysis")
    parser.add_argument("--ep_prior_ckpt", type=str,
                        default="/root/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="/root/ep-prior/runs/baseline_v1_contrastive/checkpoints/last.ckpt")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--output_dir", type=str, default="/root/ep-prior/runs/failure_mode_analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading models...")
    ep_prior, baseline = load_models(args.ep_prior_ckpt, args.baseline_ckpt, args.device)
    
    print("Loading datasets...")
    train_dataset = PTBXLDataset(args.data_path, split="train", return_labels=True)
    test_dataset = PTBXLDataset(args.data_path, split="test", return_labels=True)
    
    results_df = stratified_evaluation(
        ep_prior, baseline,
        train_dataset, test_dataset,
        device=args.device,
    )
    
    results_df.to_csv(output_dir / "failure_mode_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'failure_mode_results.csv'}")


if __name__ == "__main__":
    main()

