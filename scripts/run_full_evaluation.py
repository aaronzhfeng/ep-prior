#!/usr/bin/env python
"""
Full Evaluation Suite for EP-Prior vs Baseline

Runs:
1. Few-shot evaluation {10, 50, 100, 500} shots
2. Sample-efficiency curves
3. Concept predictability
4. Intervention selectivity tests
"""

import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.models.baseline_model import BaselineSSL
from ep_prior.data import PTBXLDataset
from ep_prior.eval.fewshot import FewShotEvaluator
from ep_prior.eval.concept_predictability import ConceptPredictabilityEvaluator, run_concept_evaluation
from ep_prior.eval.intervention import InterventionTester


def load_models(ep_prior_ckpt: str, baseline_ckpt: str, device: str = "cuda"):
    """Load both trained models."""
    print("Loading models...")
    
    # PyTorch 2.6+ requires weights_only=False for custom classes
    import torch
    torch.serialization.add_safe_globals([])  # Clear any issues
    
    ep_prior = EPPriorSSL.load_from_checkpoint(
        ep_prior_ckpt, map_location=device, weights_only=False
    )
    ep_prior.eval()
    ep_prior.to(device)
    print(f"  EP-Prior loaded from {ep_prior_ckpt}")
    
    baseline = BaselineSSL.load_from_checkpoint(
        baseline_ckpt, map_location=device, weights_only=False
    )
    baseline.eval()
    baseline.to(device)
    print(f"  Baseline loaded from {baseline_ckpt}")
    
    return ep_prior, baseline


def run_fewshot_evaluation(
    ep_prior, baseline, 
    train_dataset, test_dataset,
    shot_sizes=[10, 50, 100, 500],
    num_seeds=3,
    device="cuda"
):
    """Run few-shot evaluation on both models."""
    print("\n" + "="*60)
    print("FEW-SHOT EVALUATION")
    print("="*60)
    
    results = {}
    
    # EP-Prior few-shot
    print("\nEvaluating EP-Prior...")
    ep_evaluator = FewShotEvaluator(
        ep_prior, train_dataset, test_dataset,
        shot_sizes=shot_sizes, num_seeds=num_seeds, device=device
    )
    ep_results = ep_evaluator.evaluate_all(embedding_keys=["concat"])
    ep_summary = ep_evaluator.summarize_results(ep_results)
    results["ep_prior"] = ep_results
    print(ep_summary)
    
    # Baseline few-shot  
    print("\nEvaluating Baseline...")
    baseline_evaluator = FewShotEvaluator(
        baseline, train_dataset, test_dataset,
        shot_sizes=shot_sizes, num_seeds=num_seeds, device=device
    )
    baseline_results = baseline_evaluator.evaluate_all(embedding_keys=["concat"])
    baseline_summary = baseline_evaluator.summarize_results(baseline_results)
    results["baseline"] = baseline_results
    print(baseline_summary)
    
    return results


def run_concept_eval(ep_prior, train_dataset, test_dataset, device="cuda"):
    """Run concept predictability evaluation (EP-Prior only)."""
    print("\n" + "="*60)
    print("CONCEPT PREDICTABILITY EVALUATION")
    print("="*60)
    
    results, summary = run_concept_evaluation(ep_prior, train_dataset, test_dataset, device=device)
    return results, summary


def run_intervention_tests(ep_prior, test_dataset, device="cuda"):
    """Run intervention selectivity tests (EP-Prior only)."""
    print("\n" + "="*60)
    print("INTERVENTION SELECTIVITY TESTS")
    print("="*60)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    tester = InterventionTester(ep_prior, device=device)
    results = tester.evaluate_all_interventions(loader, n_base_samples=20)
    
    print("\nLeakage Metrics (lower is better, target: <10%):")
    for component, leakage in results.items():
        if isinstance(leakage, dict) and "mean_leakage" in leakage:
            print(f"  {component}: {leakage['mean_leakage']*100:.1f}%")
        elif isinstance(leakage, (int, float)):
            print(f"  {component}: {leakage*100:.1f}%")
    
    return results


def plot_sample_efficiency_curves(fewshot_results, save_path):
    """Generate sample efficiency curves (KEY FIGURE)."""
    print("\n" + "="*60)
    print("GENERATING SAMPLE EFFICIENCY CURVES")
    print("="*60)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {"ep_prior": "#2ecc71", "baseline": "#e74c3c"}
    labels = {"ep_prior": "EP-Prior", "baseline": "Baseline"}
    
    for model_name, results_df in fewshot_results.items():
        # Group by shot size
        grouped = results_df.groupby("shot_size").agg({
            "auroc_macro": ["mean", "std"]
        }).reset_index()
        grouped.columns = ["shot_size", "auroc_mean", "auroc_std"]
        
        ax.errorbar(
            grouped["shot_size"], 
            grouped["auroc_mean"],
            yerr=grouped["auroc_std"],
            marker="o", 
            markersize=8,
            linewidth=2,
            capsize=5,
            label=labels[model_name],
            color=colors[model_name]
        )
    
    ax.set_xlabel("Number of Training Samples per Class", fontsize=12)
    ax.set_ylabel("AUROC (macro)", fontsize=12)
    ax.set_title("Sample Efficiency: EP-Prior vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


def generate_results_table(fewshot_results, concept_results=None):
    """Generate results table for paper."""
    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    
    # Few-shot results
    print("\n### Few-Shot AUROC (macro)")
    print("| Model | 10-shot | 50-shot | 100-shot | 500-shot |")
    print("|-------|---------|---------|----------|----------|")
    
    for model_name, results_df in fewshot_results.items():
        grouped = results_df.groupby("shot_size")["auroc_macro"].agg(["mean", "std"])
        row = f"| {model_name.replace('_', '-').title()} |"
        for shot in [10, 50, 100, 500]:
            if shot in grouped.index:
                mean, std = grouped.loc[shot, "mean"], grouped.loc[shot, "std"]
                row += f" {mean:.3f}±{std:.3f} |"
            else:
                row += " - |"
        print(row)
    
    # Concept predictability (if available)
    if concept_results is not None and len(concept_results) > 0:
        print("\n### Concept Predictability (AUROC)")
        print("| Latent | CD | HYP | MI | NORM | STTC |")
        print("|--------|-----|-----|-----|------|------|")
        
        auroc_table = concept_results.get("auroc_table")
        if auroc_table is not None:
            for emb in ["P", "QRS", "T", "HRV", "concat"]:
                if emb in auroc_table.columns.get_level_values(0):
                    row = f"| z_{emb} |"
                    for cls in ["CD", "HYP", "MI", "NORM", "STTC"]:
                        if cls in auroc_table.index:
                            val = auroc_table.loc[cls, emb] if emb in auroc_table.columns else "-"
                            row += f" {val:.3f} |" if isinstance(val, float) else f" {val} |"
                    print(row)


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--ep_prior_ckpt", type=str, 
                        default="/root/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="/root/ep-prior/runs/baseline_v1_contrastive/checkpoints/last.ckpt")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. If not specified, creates timestamped folder.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--skip_intervention", action="store_true", help="Skip intervention tests")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/root/ep-prior/runs/evaluation_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EP-PRIOR FULL EVALUATION SUITE")
    print("="*60)
    print(f"EP-Prior checkpoint: {args.ep_prior_ckpt}")
    print(f"Baseline checkpoint: {args.baseline_ckpt}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # Load models
    ep_prior, baseline = load_models(args.ep_prior_ckpt, args.baseline_ckpt, args.device)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PTBXLDataset(args.data_path, split="train", return_labels=True)
    test_dataset = PTBXLDataset(args.data_path, split="test", return_labels=True)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # 1. Few-shot evaluation
    fewshot_results = run_fewshot_evaluation(
        ep_prior, baseline,
        train_dataset, test_dataset,
        shot_sizes=[10, 50, 100, 500],
        num_seeds=args.num_seeds,
        device=args.device
    )
    
    # Save few-shot results
    for model_name, df in fewshot_results.items():
        df.to_csv(output_dir / f"fewshot_{model_name}.csv", index=False)
    
    # 2. Sample efficiency curves
    plot_sample_efficiency_curves(
        fewshot_results, 
        output_dir / "sample_efficiency_curve.png"
    )
    
    # 3. Concept predictability (EP-Prior only)
    concept_results, concept_summary = run_concept_eval(
        ep_prior, train_dataset, test_dataset, args.device
    )
    
    # 4. Intervention tests (EP-Prior only)
    intervention_results = None
    if not args.skip_intervention:
        intervention_results = run_intervention_tests(ep_prior, test_dataset, args.device)
    
    # 5. Generate results table
    generate_results_table(fewshot_results, concept_results)
    
    # Save all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "ep_prior_ckpt": args.ep_prior_ckpt,
        "baseline_ckpt": args.baseline_ckpt,
        "fewshot_summary": {
            model: df.groupby("shot_size")["auroc_macro"].agg(["mean", "std"]).to_dict()
            for model, df in fewshot_results.items()
        },
    }
    
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

