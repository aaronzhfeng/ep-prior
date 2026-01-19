#!/usr/bin/env python
"""
Ablation Study: EP-Prior with vs without EP Constraints

Trains EP-Prior model but disables EP constraint losses to isolate
the contribution of the soft EP priors.

Model variants:
- EP-Prior (full): recon + contrastive + EP constraints
- EP-Prior (no EP): recon + contrastive only (λ_ep = 0)

This allows testing whether the EP constraints are necessary for
the sample efficiency gains, or if structured latent space alone suffices.
"""

import argparse
import torch
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataModule


def train_ablation_no_ep(args):
    """
    Train EP-Prior with EP constraints disabled (lambda_ep = 0).
    """
    print("="*60)
    print("ABLATION: EP-Prior WITHOUT EP Constraints")
    print("="*60)
    
    # Create data module
    data_module = PTBXLDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create model with EP constraints disabled
    model = EPPriorSSL(
        input_channels=12,
        backbone_name="xresnet1d50",
        d_P=32,
        d_QRS=128,
        d_T=64,
        d_HRV=32,
        n_leads_decoder=12,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lam_recon=1.0,
        lam_ep=0.0,  # DISABLED for ablation
        lam_contrast=args.lambda_contrastive,
    )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_no_ep_{timestamp}"
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=run_name,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.log_dir) / run_name / "checkpoints",
        filename="epoch={epoch}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
    )
    
    print(f"\nConfig:")
    print(f"  lambda_recon: 1.0")
    print(f"  lambda_ep: 0.0 (DISABLED)")
    print(f"  lambda_contrastive: {args.lambda_contrastive}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_epochs: {args.max_epochs}")
    print(f"  Run name: {run_name}")
    
    # Train
    trainer.fit(model, data_module)
    
    print(f"\nTraining complete! Checkpoint saved to:")
    print(f"  {checkpoint_callback.best_model_path}")
    
    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Ablation: Train EP-Prior without EP constraints")
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--log_dir", type=str, default="/root/ep-prior/runs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    
    args = parser.parse_args()
    
    train_ablation_no_ep(args)


if __name__ == "__main__":
    main()

