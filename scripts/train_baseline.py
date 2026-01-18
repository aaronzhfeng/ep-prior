#!/usr/bin/env python
"""
Train Capacity-Matched Baseline

Same architecture capacity as EP-Prior but without:
- Structured latent space
- EP constraints
- Gaussian wave decoder

This provides a fair comparison to isolate the effect of EP structure.
"""

import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import sys
sys.path.insert(0, "/root/ep-prior")

from ep_prior.models.baseline_model import BaselineSSL
from ep_prior.data import PTBXLDataModule, ECGAugmentation


def main():
    parser = argparse.ArgumentParser(description="Train Capacity-Matched Baseline")
    
    # Data
    parser.add_argument("--data_path", type=str, default="/root/ep-prior/data/ptb-xl")
    parser.add_argument("--sampling_rate", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model
    parser.add_argument("--backbone", type=str, default="xresnet1d50")
    parser.add_argument("--d_latent", type=int, default=256)
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lam_recon", type=float, default=1.0)
    parser.add_argument("--lam_contrast", type=float, default=0.0)
    
    # Experiment
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ep-prior")
    parser.add_argument("--fast_dev_run", action="store_true")
    
    args = parser.parse_args()
    
    # Seed
    pl.seed_everything(args.seed)
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 60)
    print("Capacity-Matched Baseline Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Data path: {args.data_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Latent dim: {args.d_latent}")
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
    model = BaselineSSL(
        input_channels=12,
        backbone_name=args.backbone,
        d_latent=args.d_latent,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lam_recon=args.lam_recon,
        lam_contrast=args.lam_contrast,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"runs/{args.experiment_name}/checkpoints",
            filename="baseline-{epoch:02d}-{val/loss:.4f}",
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

