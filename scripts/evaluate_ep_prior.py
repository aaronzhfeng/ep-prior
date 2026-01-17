#!/usr/bin/env python
"""
Evaluate Trained EP-Prior Model

Runs the three key evaluation suites:
1. Few-shot classification (sample efficiency)
2. Concept predictability (structured latent → pathology)
3. Intervention selectivity (disentanglement)

Usage:
    python scripts/evaluate_ep_prior.py --checkpoint path/to/checkpoint.ckpt
"""

import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ecg-selfsupervised"))

import torch
import pandas as pd

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataset
from ep_prior.eval import (
    run_fewshot_evaluation,
    run_intervention_evaluation,
    run_concept_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EP-Prior model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl",
                        help="Path to PTB-XL dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    
    # Evaluation options
    parser.add_argument("--skip_fewshot", action="store_true",
                        help="Skip few-shot evaluation")
    parser.add_argument("--skip_intervention", action="store_true",
                        help="Skip intervention evaluation")
    parser.add_argument("--skip_concept", action="store_true",
                        help="Skip concept predictability")
    
    # Few-shot options
    parser.add_argument("--shot_sizes", type=int, nargs="+",
                        default=[10, 50, 100, 500],
                        help="Shot sizes for few-shot evaluation")
    parser.add_argument("--num_seeds", type=int, default=3,
                        help="Number of seeds per shot size")
    
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for evaluation")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cpu") -> EPPriorSSL:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    model = EPPriorSSL.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "runs", 
            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EP-Prior Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PTBXLDataset(
        args.data_path, split="train", 
        normalize=True, return_labels=True
    )
    test_dataset = PTBXLDataset(
        args.data_path, split="test",
        normalize=True, return_labels=True
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    results = {}
    
    # ===== 1. Few-shot evaluation =====
    if not args.skip_fewshot:
        print("\n" + "=" * 60)
        print("1. FEW-SHOT EVALUATION")
        print("=" * 60)
        
        fewshot_results, fewshot_summary = run_fewshot_evaluation(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            shot_sizes=args.shot_sizes,
            num_seeds=args.num_seeds,
            device=args.device,
            save_path=os.path.join(args.output_dir, "fewshot_results.csv"),
        )
        
        print("\nFew-shot summary:")
        print(fewshot_summary)
        
        results["fewshot"] = fewshot_results
    
    # ===== 2. Concept predictability =====
    if not args.skip_concept:
        print("\n" + "=" * 60)
        print("2. CONCEPT PREDICTABILITY")
        print("=" * 60)
        
        from torch.utils.data import DataLoader
        
        concept_results, concept_summary = run_concept_evaluation(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=args.device,
            save_path=os.path.join(args.output_dir, "concept_results.csv"),
        )
        
        results["concept"] = concept_results
    
    # ===== 3. Intervention evaluation =====
    if not args.skip_intervention:
        print("\n" + "=" * 60)
        print("3. INTERVENTION SELECTIVITY")
        print("=" * 60)
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        intervention_results = run_intervention_evaluation(
            model=model,
            dataloader=test_loader,
            n_samples=10,
            device=args.device,
        )
        
        results["intervention"] = intervention_results
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("EP-Prior Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        
        if "fewshot" in results:
            f.write("Few-shot AUROC (macro):\n")
            for _, row in results["fewshot"].groupby("shot_size").mean().iterrows():
                f.write(f"  {row.name}-shot: {row['auroc_macro']:.4f}\n")
        
        if "intervention" in results:
            f.write("\nIntervention Leakage:\n")
            for target, metrics in results["intervention"].items():
                f.write(f"  {target}: mean={metrics['mean_leakage']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    main()

