#!/usr/bin/env python
"""
Train EP-Prior Self-Supervised Model

Usage:
    python scripts/train_ep_prior.py [OPTIONS]

Examples:
    # Basic training
    python scripts/train_ep_prior.py
    
    # With contrastive learning
    python scripts/train_ep_prior.py --lam_contrast 0.1
    
    # Quick test run
    python scripts/train_ep_prior.py --fast_dev_run
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ecg-selfsupervised"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataModule, ECGAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Train EP-Prior SSL model")
    
    # Data
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl",
                        help="Path to PTB-XL dataset")
    parser.add_argument("--sampling_rate", type=int, default=100,
                        help="ECG sampling rate (100 or 500)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # Model
    parser.add_argument("--backbone", type=str, default="xresnet1d50",
                        choices=["xresnet1d18", "xresnet1d34", "xresnet1d50"],
                        help="Encoder backbone")
    parser.add_argument("--d_P", type=int, default=32, help="P-wave latent dim")
    parser.add_argument("--d_QRS", type=int, default=128, help="QRS latent dim")
    parser.add_argument("--d_T", type=int, default=64, help="T-wave latent dim")
    parser.add_argument("--d_HRV", type=int, default=32, help="HRV latent dim")
    
    # Loss weights
    parser.add_argument("--lam_recon", type=float, default=1.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--lam_ep", type=float, default=0.5,
                        help="EP constraint loss weight")
    parser.add_argument("--lam_contrast", type=float, default=0.0,
                        help="Contrastive loss weight (0 to disable)")
    parser.add_argument("--ep_warmup_epochs", type=int, default=10,
                        help="Epochs for EP constraint warmup")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    
    # Logging
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="ep-prior",
                        help="W&B project name")
    
    # Debug
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Quick test run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"ep_prior_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 60)
    print("EP-Prior Self-Supervised Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Data path: {args.data_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Contrastive: {'Yes' if args.lam_contrast > 0 else 'No'}")
    print("=" * 60)
    
    # Setup augmentations for contrastive learning
    train_transform = None
    if args.lam_contrast > 0:
        train_transform = ECGAugmentation(
            time_shift_max=50,
            amplitude_scale_range=(0.8, 1.2),
            noise_std=0.05,
            lead_dropout_prob=0.1,
        )
    
    # Data module
    dm = PTBXLDataModule(
        data_path=args.data_path,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        train_transform=train_transform,
        return_labels=False,
    )
    
    # Model
    model = EPPriorSSL(
        input_channels=12,
        backbone_name=args.backbone,
        d_P=args.d_P,
        d_QRS=args.d_QRS,
        d_T=args.d_T,
        d_HRV=args.d_HRV,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lam_recon=args.lam_recon,
        lam_ep=args.lam_ep,
        lam_contrast=args.lam_contrast,
        ep_warmup_epochs=args.ep_warmup_epochs,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"runs/{args.experiment_name}/checkpoints",
            filename="ep_prior-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
        ),
    ]
    
    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir=f"runs/{args.experiment_name}",
        )
    else:
        logger = TensorBoardLogger(
            save_dir="runs",
            name=args.experiment_name,
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, dm)
    
    print("\nTraining complete!")
    print(f"Best model: {callbacks[0].best_model_path}")
    
    return model, trainer


if __name__ == "__main__":
    main()

