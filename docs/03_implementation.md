# Implementation Guide

## Overview

This document describes the codebase structure, key files, and how components interact. Use this as a map when navigating the code.

## Directory Structure

```
ep-prior/
├── ep_prior/                    # Main Python package
│   ├── __init__.py
│   ├── models/                  # Neural network architectures
│   │   ├── __init__.py
│   │   ├── structured_encoder.py    # EP-Prior encoder with attention
│   │   ├── gaussian_wave_decoder.py # Physics-informed decoder
│   │   ├── lightning_module.py      # PyTorch Lightning training module
│   │   └── baseline_model.py        # Capacity-matched baseline
│   ├── losses/                  # Loss functions
│   │   ├── __init__.py
│   │   └── ep_constraints.py        # Soft EP constraint losses
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── ptbxl_dataset.py         # PTB-XL dataset class
│   └── eval/                    # Evaluation modules
│       ├── __init__.py
│       ├── fewshot.py               # Few-shot evaluation
│       ├── probes.py                # Linear probe utilities
│       ├── concept_predictability.py # Concept-to-latent evaluation
│       └── intervention.py          # Intervention selectivity tests
├── scripts/                     # Executable scripts
│   ├── train_ep_prior.py            # Train EP-Prior model
│   ├── train_baseline.py            # Train baseline model
│   ├── train_ablation.py            # Train EP-Prior without EP loss
│   ├── evaluate_ep_prior.py         # Run evaluation suite
│   ├── run_full_evaluation.py       # Complete evaluation pipeline
│   ├── eval_failure_modes.py        # Stratified evaluation by rhythm
│   ├── eval_ablation.py             # Ablation study evaluation
│   └── generate_paper_figures.py    # Generate publication figures
├── runs/                        # Experiment outputs (gitignored)
│   ├── ep_prior_v4_contrastive_fixed/
│   │   └── checkpoints/
│   │       └── last.ckpt
│   ├── baseline_v1_contrastive/
│   │   └── checkpoints/
│   │       └── last.ckpt
│   └── evaluation_*/            # Evaluation results
├── data/                        # Dataset directory (gitignored)
│   └── ptb-xl/                  # PTB-XL dataset
├── docs/                        # This documentation
├── venv/                        # Python virtual environment
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

## Key Files Deep Dive

### 1. `ep_prior/models/structured_encoder.py`

**Purpose**: Extracts structured latent representations from ECG signals.

**Key Classes**:
- `AttentionPooling`: Learns to attend to specific temporal regions
- `StructuredEncoder`: Main encoder combining backbone + attention + projections

**Important Methods**:
```python
class StructuredEncoder(nn.Module):
    def forward(self, x, return_attention=True, return_features=False):
        """
        Args:
            x: (B, 12, T) input ECG
            return_attention: whether to return attention weights
            return_features: whether to return backbone features
        
        Returns:
            z: dict with keys ["P", "QRS", "T", "HRV"], each (B, d_*)
            attn: dict with attention weights for visualization
        """
    
    def get_latent_concat(self, z):
        """Concatenate all latent components into single vector."""
        return torch.cat([z["P"], z["QRS"], z["T"], z["HRV"]], dim=-1)
```

### 2. `ep_prior/models/gaussian_wave_decoder.py`

**Purpose**: Reconstructs ECG from structured latents using Gaussian waves.

**Key Classes**:
- `GaussianWaveDecoder`: Main decoder class

**Important Methods**:
```python
class GaussianWaveDecoder(nn.Module):
    def forward(self, z, T, return_components=False):
        """
        Args:
            z: dict with latent vectors
            T: output sequence length
            return_components: whether to return individual waves
        
        Returns:
            x_hat: (B, n_leads, T) reconstructed ECG
            params: dict with decoder parameters for EP loss
                {
                    "P": {"amp": ..., "tau": ..., "sig": ..., "gate": ...},
                    "QRS": {...},
                    "T": {...}
                }
        """
```

**Critical Implementation Details**:
- `gate_min=0.1`: Prevents gate collapse to zero
- `output_scale`: Learnable scalar initialized to 15.0
- Amplitude biases initialized to [1.0, 2.0, 1.0] for [P, QRS, T]

### 3. `ep_prior/models/lightning_module.py`

**Purpose**: PyTorch Lightning module for training.

**Key Classes**:
- `EPPriorSSL`: Main training module

**Important Methods**:
```python
class EPPriorSSL(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # Forward pass, loss computation, logging
    
    def validation_step(self, batch, batch_idx):
        # Validation metrics
    
    def configure_optimizers(self):
        # AdamW with cosine annealing
    
    def compute_contrastive_loss(self, z1, z2):
        # NT-Xent loss implementation
```

### 4. `ep_prior/losses/ep_constraints.py`

**Purpose**: Soft constraint losses encoding electrophysiology knowledge.

**Key Functions**:
```python
def ep_constraint_loss(params, config=None):
    """
    Compute total EP constraint loss.
    
    Args:
        params: decoder parameters dict
        config: constraint hyperparameters
    
    Returns:
        total_loss: scalar tensor
        breakdown: dict with individual loss components
    """

def ordering_loss(params, margin=0.05):
    """Penalize if P not before QRS not before T."""

def refractory_loss(params, min_PR=0.12, min_RT=0.2):
    """Penalize if intervals too short."""

def duration_loss(params, bounds):
    """Penalize if wave widths outside bounds."""
```

### 5. `ep_prior/data/ptbxl_dataset.py`

**Purpose**: Load and preprocess PTB-XL dataset.

**Key Classes**:
- `PTBXLDataset`: PyTorch Dataset for PTB-XL
- `PTBXLDataModule`: Lightning DataModule

**Important Methods**:
```python
class PTBXLDataset(Dataset):
    def __init__(self, data_path, split="train", return_labels=True):
        """
        Args:
            data_path: path to PTB-XL directory
            split: "train", "val", or "test"
            return_labels: whether to return diagnostic labels
        """
    
    def __getitem__(self, idx):
        """
        Returns:
            {
                "x": (12, 1000) ECG tensor,
                "superclass": (5,) multi-hot labels (if return_labels),
                "record_id": str
            }
        """
```

### 6. `ep_prior/eval/fewshot.py`

**Purpose**: Few-shot learning evaluation.

**Key Classes**:
- `FewShotEvaluator`: Runs few-shot experiments

**Important Methods**:
```python
class FewShotEvaluator:
    def evaluate_all(self, embeddings, labels, shot_sizes, n_seeds=3):
        """
        Run few-shot evaluation across multiple shot sizes and seeds.
        
        Returns:
            DataFrame with columns [shot_size, seed, condition, auroc, auprc]
        """
```

### 7. `ep_prior/eval/intervention.py`

**Purpose**: Test disentanglement via latent interventions.

**Key Classes**:
- `InterventionTester`: Runs intervention experiments

**Important Methods**:
```python
class InterventionTester:
    def run_intervention(self, base_z, target_component, direction, n_steps, scale):
        """
        Vary one latent component and observe decoder parameter changes.
        
        Returns:
            {
                "params": decoder params at each step,
                "reconstructions": x_hat at each step,
                "alphas": interpolation coefficients
            }
        """
    
    def compute_leakage(self, results, target):
        """
        Compute leakage = change in non-target / change in target.
        Low leakage (<10%) indicates good disentanglement.
        """
```

## Dependencies

### Core Dependencies (requirements.txt)

```
torch>=2.0.0
pytorch-lightning>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
wfdb>=4.1.0              # For reading ECG files
tensorboard>=2.13.0      # For logging
```

### External Package: clinical-ts

The `xresnet1d` backbone comes from the `clinical-ts` package:

```bash
# Installed from: https://github.com/broadinstitute/ecg-selfsupervised
pip install -e /path/to/ecg-selfsupervised
```

## Configuration

### Training Hyperparameters

Default values in `EPPriorSSL.__init__`:

```python
# Architecture
input_channels = 12
backbone_name = "xresnet1d50"
d_P = d_QRS = d_T = d_HRV = 64

# Loss weights
lambda_recon = 1.0
lambda_ep = 0.1
lambda_contrastive = 0.1

# Optimizer
lr = 1e-3
weight_decay = 1e-4

# Training
batch_size = 64
max_epochs = 100
```

### EP Constraint Hyperparameters

Default values in `ep_constraint_loss`:

```python
# Ordering
order_margin = 0.05  # Minimum separation between waves

# Refractory periods (in normalized time [0,1])
min_PR = 0.12  # ~120ms at 1000 samples
min_RT = 0.20  # ~200ms

# Duration bounds (sigma values)
bounds = {
    "P": (0.02, 0.08),    # 20-80ms
    "QRS": (0.02, 0.12),  # 20-120ms (wider for BBB)
    "T": (0.04, 0.15),    # 40-150ms
}
```

## Common Operations

### Load a trained model

```python
from ep_prior.models import EPPriorSSL

model = EPPriorSSL.load_from_checkpoint(
    "runs/ep_prior_v4_contrastive_fixed/checkpoints/last.ckpt",
    map_location="cuda",
    weights_only=False
)
model.eval()
```

### Extract embeddings

```python
with torch.no_grad():
    z, attn = model.encoder(x, return_attention=True)
    z_concat = model.encoder.get_latent_concat(z)
    # z_concat: (B, 256) - concatenated latents
```

### Reconstruct ECG

```python
with torch.no_grad():
    z, _ = model.encoder(x)
    x_hat, params = model.decoder(z, T=1000, return_components=True)
    # x_hat: (B, 12, 1000) - reconstructed ECG
    # params: dict with wave parameters
```

### Access individual components

```python
# Get just z_QRS
z_qrs = z["QRS"]  # (B, 64)

# Get QRS decoder parameters
qrs_params = params["QRS"]
qrs_timing = qrs_params["tau"]  # (B, 1) - normalized timing
qrs_width = qrs_params["sig"]   # (B, 1) - normalized width
qrs_amp = qrs_params["amp"]     # (B, 12) - amplitude per lead
```

## Debugging Tips

### Check decoder output scale
```python
# If reconstructions are too small/large
print(f"Input std: {x.std():.3f}")
print(f"Output std: {x_hat.std():.3f}")
print(f"Output scale: {model.decoder.output_scale.item():.3f}")
```

### Check gate values
```python
# If gates collapse to 0, model outputs flat signals
for wave in ["P", "QRS", "T"]:
    print(f"{wave} gate: {params[wave]['gate'].mean():.3f}")
```

### Visualize attention
```python
import matplotlib.pyplot as plt

# attn["P"]: (B, L) attention weights
plt.plot(attn["P"][0].cpu().numpy())
plt.title("P-wave attention")
```

## Testing

### Run unit tests (if available)
```bash
cd /root/ep-prior
python -m pytest tests/ -v
```

### Quick sanity check
```python
# Test forward pass
x = torch.randn(4, 12, 1000)
z, attn = model.encoder(x)
x_hat, params = model.decoder(z, 1000)

assert x_hat.shape == x.shape
assert all(k in z for k in ["P", "QRS", "T", "HRV"])
print("Forward pass OK!")
```

