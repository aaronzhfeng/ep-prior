# Model Architecture

## Overview

EP-Prior consists of three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         EP-Prior SSL                             │
├─────────────────────────────────────────────────────────────────┤
│  Input ECG (12-lead, 1000 samples @ 100Hz)                      │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Structured Encoder               │                    │
│  │  ┌─────────────────────────────────┐    │                    │
│  │  │   xresnet1d50 Backbone          │    │                    │
│  │  │   (12 → 512 channels)           │    │                    │
│  │  └─────────────────────────────────┘    │                    │
│  │              │                           │                    │
│  │    ┌────────┼────────┬────────┐         │                    │
│  │    ▼        ▼        ▼        ▼         │                    │
│  │  Attn_P  Attn_QRS  Attn_T  GlobalPool   │                    │
│  │    │        │        │        │         │                    │
│  │    ▼        ▼        ▼        ▼         │                    │
│  │   z_P     z_QRS     z_T     z_HRV       │                    │
│  │  (64D)    (64D)    (64D)    (64D)       │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │       Gaussian Wave Decoder              │                    │
│  │  z_P → (A_P, τ_P, σ_P, gate_P)          │                    │
│  │  z_QRS → (A_QRS, τ_QRS, σ_QRS, gate_QRS)│                    │
│  │  z_T → (A_T, τ_T, σ_T, gate_T)          │                    │
│  │                                          │                    │
│  │  x_hat = Σ gate_i · A_i · G(t; τ_i, σ_i)│                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  Reconstructed ECG (12-lead, 1000 samples)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Structured Encoder

**File**: `ep_prior/models/structured_encoder.py`

### Backbone: xresnet1d50

We use `xresnet1d50` from the `clinical-ts` library, a 1D adaptation of ResNet designed for physiological time series:

```python
from clinical_ts.xresnet1d import xresnet1d50

backbone = xresnet1d50(
    input_channels=12,  # 12-lead ECG
    num_classes=512,    # Feature dimension
)
```

**Architecture details**:
- Input: (B, 12, 1000) - Batch × Leads × Time
- Output: (B, 512, L) - Batch × Features × Reduced_Time
- L ≈ 32 after pooling layers

### Attention Pooling

Each wave component (P, QRS, T) has a dedicated attention module that learns to focus on relevant temporal regions:

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        self.query = nn.Linear(d_in, d_out)
        self.key = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)
    
    def forward(self, x):  # x: (B, D, L)
        Q = self.query(x.mean(dim=-1))  # (B, d_out)
        K = self.key(x.transpose(1, 2))  # (B, L, d_out)
        V = self.value(x.transpose(1, 2))  # (B, L, d_out)
        
        attn = softmax(Q @ K.T / sqrt(d_out))  # (B, L)
        output = (attn @ V).squeeze()  # (B, d_out)
        return output, attn
```

### Latent Projections

After attention pooling, each component is projected to its final dimension:

```python
self.proj_P = nn.Sequential(
    nn.Linear(feat_dim, feat_dim),
    nn.ReLU(),
    nn.Linear(feat_dim, d_P),  # d_P = 64
)
# Similar for proj_QRS, proj_T, proj_HRV
```

### Output Format

The encoder returns a dictionary of latent vectors:

```python
z = {
    "P": tensor(B, 64),
    "QRS": tensor(B, 64),
    "T": tensor(B, 64),
    "HRV": tensor(B, 64),
}
```

## 2. Gaussian Wave Decoder

**File**: `ep_prior/models/gaussian_wave_decoder.py`

### Core Idea

Instead of a generic MLP, the decoder reconstructs ECGs as a sum of Gaussian-shaped waves:

```
x_hat(t) = scale · Σ_{wave ∈ {P, QRS, T}} gate_wave · A_wave · G(t; τ_wave, σ_wave)
```

Where G is a Gaussian:
```
G(t; τ, σ) = exp(-(t - τ)² / (2σ²))
```

### Parameter Networks

Each wave component has MLPs that predict its parameters:

```python
# For each wave (P, QRS, T):
self.amp_P = MLP(d_P, hidden, n_leads)      # Amplitude per lead
self.tau_P = MLP(d_P, hidden, 1)            # Timing (0-1, normalized)
self.sig_P = MLP(d_P, hidden, 1)            # Width (0-1, normalized)
self.gate_P = MLP(d_P, hidden, 1)           # Presence gate (0-1)
```

### Parameter Constraints

Parameters are constrained to physiologically reasonable ranges:

```python
# Amplitude: unbounded (can be positive or negative)
A = self.amp_P(z_P)  # (B, n_leads)

# Timing: sigmoid to [0, 1], then scale to [0, T]
tau = torch.sigmoid(self.tau_P(z_P))  # (B, 1) in [0, 1]

# Width: softplus to ensure positive, then scale
sig = F.softplus(self.sig_P(z_P)) * 0.1  # (B, 1), typically 0.05-0.2

# Gate: sigmoid with minimum value to prevent collapse
gate = torch.sigmoid(self.gate_P(z_P)).clamp(min=0.1)  # (B, 1) in [0.1, 1]
```

### Wave Generation

Each wave is generated as a Gaussian pulse:

```python
def _generate_wave(self, A, tau, sig, gate, T):
    t = torch.linspace(0, 1, T)  # (T,)
    
    # Gaussian centered at tau with width sig
    gaussian = torch.exp(-0.5 * ((t - tau) / sig) ** 2)  # (B, T)
    
    # Scale by amplitude and gate
    wave = gate * A.unsqueeze(-1) * gaussian.unsqueeze(1)  # (B, n_leads, T)
    
    return wave
```

### Final Reconstruction

```python
def forward(self, z, T):
    x_P = self._generate_wave(A_P, tau_P, sig_P, gate_P, T)
    x_QRS = self._generate_wave(A_QRS, tau_QRS, sig_QRS, gate_QRS, T)
    x_T = self._generate_wave(A_T, tau_T, sig_T, gate_T, T)
    
    x_hat = (x_P + x_QRS + x_T) * self.output_scale
    
    return x_hat, params
```

### Output Scale

A learnable scalar `output_scale` (initialized to 15.0) allows the model to match the amplitude range of real ECGs:

```python
self.output_scale = nn.Parameter(torch.tensor(15.0))
```

## 3. EP Constraint Loss

**File**: `ep_prior/losses/ep_constraints.py`

### Ordering Constraint

P wave must precede QRS, which must precede T:

```python
def ordering_loss(params):
    tau_P = params["P"]["tau"]
    tau_QRS = params["QRS"]["tau"]
    tau_T = params["T"]["tau"]
    
    # Penalize if P comes after QRS
    loss_P_QRS = F.relu(tau_P - tau_QRS + margin)
    
    # Penalize if QRS comes after T
    loss_QRS_T = F.relu(tau_QRS - tau_T + margin)
    
    return loss_P_QRS.mean() + loss_QRS_T.mean()
```

### Refractory Period Constraint

Minimum time between waves:

```python
def refractory_loss(params, min_PR=0.12, min_RT=0.2):
    # PR interval (P to QRS) must be at least 120ms
    PR = params["QRS"]["tau"] - params["P"]["tau"]
    loss_PR = F.relu(min_PR - PR)
    
    # RT interval (QRS to T) must be at least 200ms  
    RT = params["T"]["tau"] - params["QRS"]["tau"]
    loss_RT = F.relu(min_RT - RT)
    
    return loss_PR.mean() + loss_RT.mean()
```

### Duration Bounds Constraint

Wave widths must be within physiological limits:

```python
def duration_loss(params, bounds):
    total = 0
    for wave in ["P", "QRS", "T"]:
        sig = params[wave]["sig"]
        sig_min, sig_max = bounds[wave]
        
        # Penalize if too narrow or too wide
        total += F.relu(sig_min - sig).mean()
        total += F.relu(sig - sig_max).mean()
    
    return total
```

## 4. Lightning Module

**File**: `ep_prior/models/lightning_module.py`

### Training Step

```python
def training_step(self, batch, batch_idx):
    x = batch["x"]  # (B, 12, 1000)
    
    # Encode
    z, attn = self.encoder(x)
    
    # Decode
    x_hat, params = self.decoder(z, x.shape[-1])
    
    # Losses
    recon_loss = F.mse_loss(x_hat, x) / (x.std() ** 2 + 1e-8)
    ep_loss = self.ep_constraint_loss(params)
    
    # Optional contrastive loss
    if self.lambda_contrastive > 0:
        x_aug = self.augment(x)
        z_aug, _ = self.encoder(x_aug)
        contrast_loss = self.contrastive_loss(z, z_aug)
    else:
        contrast_loss = 0
    
    total_loss = (
        self.lambda_recon * recon_loss +
        self.lambda_ep * ep_loss +
        self.lambda_contrastive * contrast_loss
    )
    
    return total_loss
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_P, d_QRS, d_T, d_HRV` | 64 each | Latent dimensions |
| `lambda_recon` | 1.0 | Reconstruction weight |
| `lambda_ep` | 0.1 | EP constraint weight |
| `lambda_contrastive` | 0.1 | Contrastive loss weight |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `batch_size` | 64 | Training batch size |

## 5. Baseline Model (for Comparison)

**File**: `ep_prior/models/baseline_model.py`

The baseline uses the **same backbone** but without structured latents or EP constraints:

```python
class GenericEncoder(nn.Module):
    def __init__(self, d_latent=256):  # Same total dimension
        self.backbone = xresnet1d50(input_channels=12)
        self.proj = nn.Linear(feat_dim, d_latent)
    
    def forward(self, x):
        features = self.backbone(x)
        z = self.proj(features.mean(dim=-1))  # Global pooling
        return z  # Single 256D vector (not structured)

class GenericDecoder(nn.Module):
    def __init__(self, d_latent=256, n_leads=12, T_out=1000):
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, 512),
            nn.ReLU(),
            nn.Linear(512, n_leads * T_out),
        )
    
    def forward(self, z, T):
        x_hat = self.mlp(z).view(-1, 12, T)
        return x_hat, {}  # No interpretable params
```

**Key difference**: Same capacity (256D latent), but unstructured and no EP constraints.

## 6. Model Sizes

| Component | Parameters |
|-----------|------------|
| xresnet1d50 backbone | ~23M |
| Attention modules (×3) | ~0.5M |
| Projection heads (×4) | ~0.3M |
| Gaussian wave decoder | ~0.2M |
| **Total EP-Prior** | **~24M** |
| **Total Baseline** | **~24M** |

Both models have nearly identical parameter counts for fair comparison.

