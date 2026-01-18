# Training Guide

## Overview

This document covers how to train EP-Prior and baseline models, including commands, configurations, and troubleshooting.

## Prerequisites

### 1. Environment Setup

```bash
cd /root/ep-prior
source venv/bin/activate

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Data Preparation

The PTB-XL dataset should be at `/root/ep-prior/data/ptb-xl/`. If missing:

```bash
# Download from PhysioNet S3 (fastest method)
aws s3 sync --no-sign-request \
    s3://physionet-open/ptb-xl/1.0.3/ \
    /root/ep-prior/data/ptb-xl/

# Or use wget (slower)
wget -r -N -c -np \
    https://physionet.org/files/ptb-xl/1.0.3/ \
    -P /root/ep-prior/data/
```

Expected structure:
```
data/ptb-xl/
├── ptbxl_database.csv      # Metadata
├── scp_statements.csv      # Label definitions
├── records100/             # 100Hz signals
│   ├── 00000/
│   ├── 01000/
│   └── ...
└── records500/             # 500Hz signals (optional)
```

## Training EP-Prior

### Basic Training Command

```bash
cd /root/ep-prior && source venv/bin/activate

python scripts/train_ep_prior.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-3 \
    --lambda_contrastive 0.1 \
    --lambda_ep 0.1
```

### Full Training Command with All Options

```bash
python scripts/train_ep_prior.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --log_dir /root/ep-prior/runs \
    --batch_size 64 \
    --num_workers 4 \
    --max_epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --lambda_recon 1.0 \
    --lambda_ep 0.1 \
    --lambda_contrastive 0.1 \
    --backbone xresnet1d50 \
    --d_latent 64
```

### Training with TensorBoard Logging

```bash
# Start training (logs automatically go to runs/*/lightning_logs/)
python scripts/train_ep_prior.py --max_epochs 100

# In another terminal, start TensorBoard
tensorboard --logdir /root/ep-prior/runs --port 6006

# Access at http://localhost:6006
```

### Training with Visible Progress (No Buffering)

```bash
# Option 1: Unbuffered Python output
PYTHONUNBUFFERED=1 python scripts/train_ep_prior.py --max_epochs 100

# Option 2: Using script command
script -q -c "python scripts/train_ep_prior.py --max_epochs 100" /dev/null

# Option 3: Using stdbuf
stdbuf -oL python scripts/train_ep_prior.py --max_epochs 100
```

## Training Baseline Model

For fair comparison, train a capacity-matched baseline:

```bash
python scripts/train_baseline.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --batch_size 64 \
    --max_epochs 100 \
    --lambda_contrastive 0.1
```

**Key differences from EP-Prior**:
- Same xresnet1d50 backbone
- Single 256D latent (not structured)
- MLP decoder (not Gaussian wave)
- No EP constraint loss

## Training Ablation Model

To isolate the effect of EP constraints:

```bash
python scripts/train_ablation.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --max_epochs 50 \
    --lambda_contrastive 0.1
```

This trains EP-Prior architecture but with `lambda_ep = 0`.

## Checkpoints

### Checkpoint Location

```
runs/
├── ep_prior_v4_contrastive_fixed/
│   ├── checkpoints/
│   │   ├── last.ckpt           # Most recent
│   │   └── epoch=X-val_loss=Y.ckpt  # Best by val loss
│   └── lightning_logs/
│       └── version_0/
│           └── events.out.tfevents.*
├── baseline_v1_contrastive/
│   └── checkpoints/
│       └── last.ckpt
└── ablation_no_ep_*/
    └── checkpoints/
        └── last.ckpt
```

### Loading a Checkpoint

```python
from ep_prior.models import EPPriorSSL

# Load for inference
model = EPPriorSSL.load_from_checkpoint(
    "runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt",
    map_location="cuda",
    weights_only=False
)
model.eval()

# Load for continued training
model = EPPriorSSL.load_from_checkpoint(
    "runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt"
)
trainer = pl.Trainer(max_epochs=150)
trainer.fit(model, datamodule)
```

## Hyperparameter Guide

### Loss Weights

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lambda_recon` | 1.0 | Higher = better reconstruction, may overfit |
| `lambda_ep` | 0.1 | Higher = stricter EP constraints, may hurt pathological cases |
| `lambda_contrastive` | 0.1 | Higher = better clustering, may hurt reconstruction |

### Learning Rate

| Value | Use Case |
|-------|----------|
| 1e-3 | Default, works well |
| 5e-4 | If training unstable |
| 1e-4 | Fine-tuning |

### Batch Size

| Value | Notes |
|-------|-------|
| 32 | If GPU memory limited |
| 64 | Default, good balance |
| 128 | If >24GB GPU memory |

### Epochs

| Value | Notes |
|-------|-------|
| 50 | Quick experiment |
| 100 | Full training (default) |
| 150+ | Diminishing returns |

## Monitoring Training

### Key Metrics to Watch

```
train/recon_loss    # Should decrease to ~0.3-0.5
train/ep_loss       # Should be low (~0.01-0.1), not zero
train/contrast_loss # Should decrease, stabilize ~2-3
val/recon_loss      # Should track train, not diverge
```

### Healthy Training Signs

1. **recon_loss** decreases smoothly from ~1.0 to ~0.3
2. **ep_loss** stays small but non-zero (0.01-0.1)
3. **gate values** stay in (0.1, 1.0), not collapsing
4. **output_scale** stabilizes around 10-20

### Warning Signs

| Symptom | Cause | Fix |
|---------|-------|-----|
| recon_loss stuck at 1.0 | Output too small | Check output_scale, amplitude init |
| ep_loss = 0 | Gates collapsed | Increase gate_min |
| NaN loss | Numerical instability | Lower learning rate, check contrastive loss |
| val_loss diverging | Overfitting | Add dropout, reduce epochs |

## Troubleshooting

### Issue: "CUDA out of memory"

```bash
# Reduce batch size
python scripts/train_ep_prior.py --batch_size 32

# Or use gradient accumulation
python scripts/train_ep_prior.py --batch_size 32 --accumulate_grad_batches 2
```

### Issue: Training too slow

```bash
# Increase workers
python scripts/train_ep_prior.py --num_workers 8

# Use mixed precision (if supported)
python scripts/train_ep_prior.py --precision 16
```

### Issue: "ModuleNotFoundError: No module named 'clinical_ts'"

```bash
# Install the ecg-selfsupervised package
cd /root
git clone https://github.com/broadinstitute/ecg-selfsupervised.git
pip install -e ecg-selfsupervised/
```

### Issue: Reconstruction looks wrong

```python
# Debug: Check decoder parameters
with torch.no_grad():
    z, _ = model.encoder(x)
    x_hat, params = model.decoder(z, 1000)
    
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Output range: [{x_hat.min():.2f}, {x_hat.max():.2f}]")
    print(f"Output scale: {model.decoder.output_scale.item():.2f}")
    
    for wave in ["P", "QRS", "T"]:
        print(f"{wave} gate: {params[wave]['gate'].mean():.3f}")
        print(f"{wave} tau: {params[wave]['tau'].mean():.3f}")
```

## Training History

### Successful Runs

| Run | Config | Result |
|-----|--------|--------|
| `ep_prior_v4_contrastive_fixed` | λ_ep=0.1, λ_c=0.1, 100 epochs | **Current best** |
| `baseline_v1_contrastive` | λ_c=0.1, 100 epochs | Baseline model |

### Failed Runs (for reference)

| Run | Issue | Lesson |
|-----|-------|--------|
| v1 | Gates collapsed to 0 | Added gate_min=0.1 |
| v2 | Output too small | Added output_scale parameter |
| v3_contrastive | NaN in contrastive loss | Fixed NT-Xent numerical stability |

## Estimated Training Times

| Model | GPU | Epochs | Time |
|-------|-----|--------|------|
| EP-Prior | A100 40GB | 100 | ~2 hours |
| EP-Prior | V100 16GB | 100 | ~4 hours |
| EP-Prior | RTX 3090 | 100 | ~3 hours |
| Baseline | Any | 100 | Similar to EP-Prior |

