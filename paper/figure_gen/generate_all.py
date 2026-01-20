#!/usr/bin/env python
"""
Generate All Paper Figures for EP-Prior

Master script that orchestrates generation of all figures (2-6).
Figure 1 (architecture) is manually created.

Usage:
    python generate_all.py                    # Generate all figures
    python generate_all.py --figures 2 3 5    # Generate specific figures
    python generate_all.py --output_dir /path # Custom output directory
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/root/workspace/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataset

from common import setup_style, RESULTS_DIR, OUTPUT_DIR

# Import individual figure generators
import fig2_sample_efficiency
import fig3_intervention
import fig4_ablation
import fig5_tsne
import fig6_reconstruction


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--figures", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="Which figures to generate (2-6)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures")
    parser.add_argument("--checkpoint", type=str,
                        default="/root/workspace/ep-prior/runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup directories
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EP-PRIOR PAPER FIGURE GENERATION")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Figures to generate: {args.figures}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load model and data only if needed for model-dependent figures
    model = None
    dataloader = None
    model_figures = {3, 5, 6}  # Figures that need model
    
    if model_figures & set(args.figures):
        print("\nLoading EP-Prior model...")
        model = EPPriorSSL.load_from_checkpoint(
            args.checkpoint, map_location=args.device, weights_only=False
        )
        model.eval()
        model.to(args.device)
        
        print("Loading test dataset...")
        test_dataset = PTBXLDataset(
            "/root/workspace/ep-prior/data/ptb-xl", 
            split='test', 
            return_labels=True
        )
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Generate requested figures
    for fig_num in sorted(args.figures):
        try:
            if fig_num == 2:
                fig2_sample_efficiency.generate_figure(results_dir, output_dir)
            elif fig_num == 3:
                fig3_intervention.generate_figure(model, dataloader, output_dir, args.device)
            elif fig_num == 4:
                fig4_ablation.generate_figure(results_dir, output_dir)
            elif fig_num == 5:
                fig5_tsne.generate_figure(model, dataloader, output_dir, args.device)
            elif fig_num == 6:
                fig6_reconstruction.generate_figure(model, dataloader, output_dir, args.device)
            else:
                print(f"\n[Warning] Figure {fig_num} not recognized (valid: 2-6)")
        except Exception as e:
            print(f"\n[Error] Failed to generate Figure {fig_num}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

