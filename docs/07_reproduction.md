# Reproduction Guide

## Overview

This document provides step-by-step instructions to reproduce all EP-Prior experiments from scratch, including environment setup, data download, training, and evaluation.

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 20GB | 50GB |
| CPU | 4 cores | 8+ cores |

### Software Requirements

- Linux (Ubuntu 20.04+ recommended)
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- Git

## Step 1: Clone Repository

```bash
cd /root
git clone <repository_url> ep-prior
cd ep-prior
```

Or if starting from existing code:
```bash
cd /root/ep-prior
```

## Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.10+
```

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pytorch-lightning>=2.0.0
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tqdm wfdb tensorboard
pip install scipy

# Install clinical-ts for xresnet1d backbone
cd /root
git clone https://github.com/broadinstitute/ecg-selfsupervised.git
pip install -e ecg-selfsupervised/
cd /root/ep-prior

# Install ep-prior package in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import pytorch_lightning as pl
print(f'PyTorch Lightning: {pl.__version__}')

from clinical_ts.xresnet1d import xresnet1d50
print('clinical_ts: OK')

from ep_prior.models import EPPriorSSL
print('ep_prior: OK')

print('\\nAll dependencies installed successfully!')
"
```

## Step 4: Download PTB-XL Dataset

### Option A: AWS S3 (Fastest)

```bash
# Install AWS CLI if needed
pip install awscli

# Download (no credentials needed)
aws s3 sync --no-sign-request \
    s3://physionet-open/ptb-xl/1.0.3/ \
    /root/ep-prior/data/ptb-xl/

# Verify download
ls -la /root/ep-prior/data/ptb-xl/
# Should see: ptbxl_database.csv, scp_statements.csv, records100/, records500/
```

### Option B: Wget (Slower but reliable)

```bash
mkdir -p /root/ep-prior/data
cd /root/ep-prior/data

wget -r -N -c -np \
    --accept "*.csv,*.hea,*.dat" \
    https://physionet.org/files/ptb-xl/1.0.3/

# Move files to correct location
mv physionet.org/files/ptb-xl/1.0.3 ptb-xl
rm -rf physionet.org
```

### Verify Dataset

```bash
python -c "
from ep_prior.data import PTBXLDataset

train_ds = PTBXLDataset('/root/ep-prior/data/ptb-xl', split='train')
print(f'Training samples: {len(train_ds)}')

test_ds = PTBXLDataset('/root/ep-prior/data/ptb-xl', split='test')
print(f'Test samples: {len(test_ds)}')

# Check a sample
sample = train_ds[0]
print(f'Sample shape: {sample[\"x\"].shape}')  # Should be (12, 1000)
print('Dataset loaded successfully!')
"
```

Expected output:
```
Training samples: 17418
Test samples: 2198
Sample shape: torch.Size([12, 1000])
Dataset loaded successfully!
```

## Step 5: Train EP-Prior Model

### Full Training (Recommended)

```bash
cd /root/ep-prior
source venv/bin/activate

PYTHONUNBUFFERED=1 python scripts/train_ep_prior.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --log_dir /root/ep-prior/runs \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-3 \
    --lambda_recon 1.0 \
    --lambda_ep 0.1 \
    --lambda_contrastive 0.1
```

**Expected time**: ~2-4 hours depending on GPU

### Quick Training (For testing)

```bash
python scripts/train_ep_prior.py \
    --batch_size 64 \
    --max_epochs 10
```

### Monitor Training

```bash
# In another terminal
tensorboard --logdir /root/ep-prior/runs --port 6006
# Open http://localhost:6006
```

## Step 6: Train Baseline Model

```bash
PYTHONUNBUFFERED=1 python scripts/train_baseline.py \
    --data_path /root/ep-prior/data/ptb-xl \
    --batch_size 64 \
    --max_epochs 100 \
    --lambda_contrastive 0.1
```

**Expected time**: ~2-4 hours

## Step 7: Run Evaluation

### Full Evaluation Suite

```bash
python scripts/run_full_evaluation.py \
    --ep_prior_ckpt runs/ep_prior_*/checkpoints/last.ckpt \
    --baseline_ckpt runs/baseline_*/checkpoints/last.ckpt \
    --num_seeds 3
```

**Expected time**: ~20 minutes

### Generate Paper Figures

```bash
python scripts/generate_paper_figures.py \
    --results_dir runs/evaluation_*/
```

## Step 8: Verify Results

### Expected Results

```python
# Check few-shot results
import pandas as pd

ep = pd.read_csv('runs/evaluation_*/fewshot_ep_prior.csv')
base = pd.read_csv('runs/evaluation_*/fewshot_baseline.csv')

print("10-shot AUROC:")
print(f"  EP-Prior: {ep[ep['n_shots']==10]['auroc'].mean():.3f}")
print(f"  Baseline: {base[base['n_shots']==10]['auroc'].mean():.3f}")

# Expected:
# EP-Prior: ~0.72
# Baseline: ~0.68
```

### Validate Intervention Tests

```python
# Leakage should be ~0%
python -c "
from ep_prior.eval.intervention import InterventionTester
from ep_prior.models import EPPriorSSL
from ep_prior.data import PTBXLDataset
from torch.utils.data import DataLoader

model = EPPriorSSL.load_from_checkpoint('runs/ep_prior_*/checkpoints/last.ckpt')
model.eval()
loader = DataLoader(PTBXLDataset('data/ptb-xl', 'test'), batch_size=32)

tester = InterventionTester(model, device='cuda')
results = tester.evaluate_all_interventions(loader)

for c, m in results.items():
    print(f'{c}: {m[\"mean_leakage\"]*100:.1f}%')
# Expected: All ~0%
"
```

## Complete Reproduction Script

Save this as `reproduce_all.sh`:

```bash
#!/bin/bash
set -e

echo "=== EP-Prior Full Reproduction ==="
cd /root/ep-prior
source venv/bin/activate

echo "1. Training EP-Prior..."
python scripts/train_ep_prior.py \
    --max_epochs 100 \
    --lambda_contrastive 0.1 \
    --lambda_ep 0.1

echo "2. Training Baseline..."
python scripts/train_baseline.py \
    --max_epochs 100 \
    --lambda_contrastive 0.1

echo "3. Running Evaluation..."
EP_CKPT=$(ls -t runs/ep_prior_*/checkpoints/last.ckpt | head -1)
BASE_CKPT=$(ls -t runs/baseline_*/checkpoints/last.ckpt | head -1)

python scripts/run_full_evaluation.py \
    --ep_prior_ckpt $EP_CKPT \
    --baseline_ckpt $BASE_CKPT \
    --num_seeds 3

echo "4. Generating Figures..."
RESULTS_DIR=$(ls -td runs/evaluation_*/ | head -1)
python scripts/generate_paper_figures.py \
    --results_dir $RESULTS_DIR

echo "=== Reproduction Complete ==="
echo "Results in: $RESULTS_DIR"
```

Run with:
```bash
chmod +x reproduce_all.sh
./reproduce_all.sh 2>&1 | tee reproduction.log
```

## Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size
python scripts/train_ep_prior.py --batch_size 32
```

### "ModuleNotFoundError: clinical_ts"

```bash
cd /root
git clone https://github.com/broadinstitute/ecg-selfsupervised.git
pip install -e ecg-selfsupervised/
```

### "FileNotFoundError: ptb-xl"

```bash
# Re-download dataset
aws s3 sync --no-sign-request \
    s3://physionet-open/ptb-xl/1.0.3/ \
    /root/ep-prior/data/ptb-xl/
```

### Training loss stuck at 1.0

This was fixed in v4. If using old code:
1. Check `decoder.output_scale` is initialized to ~15.0
2. Check `gate_min=0.1` in decoder
3. Check amplitude bias initialization

### NaN in contrastive loss

Fixed in v4. If using old code, update `_nt_xent_loss` to use indexing instead of mask multiplication.

## Checkpoints Available

If you want to skip training, use pre-trained checkpoints:

```
runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt  # EP-Prior
runs/baseline_v1_contrastive/checkpoints/last.ckpt        # Baseline
```

## Timeline Estimate

| Step | Time |
|------|------|
| Environment setup | 15 min |
| Data download | 10-30 min |
| EP-Prior training | 2-4 hours |
| Baseline training | 2-4 hours |
| Evaluation | 20 min |
| Paper figures | 10 min |
| **Total** | **5-9 hours** |

With pre-trained checkpoints: ~30 minutes

